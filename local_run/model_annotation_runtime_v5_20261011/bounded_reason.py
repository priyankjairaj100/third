"""Prospective v5 engineering decoder; no change to frozen v2-v4 evidence.

Immutable lexical guard intersects the original LMFE JSON schema with exactly
len(decoded_reason.split()) <= 80. This is a decoding constraint, not a label edit.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from lmformatenforcer import CharacterLevelParser, CharacterLevelParserConfig, JsonSchemaParser

WORD_LIMIT = 80
TOKEN_POLICY = {
    'schema': 'third-standalone-token-policy-v5',
    'standalone_decode': {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False},
    'regular_token_must_decode_nonempty': True,
    'regular_token_must_not_contain': 'U+FFFD',
    'eos_policy': 'Separate EOS IDs; permitted only after complete guarded JSON',
    'non_eos_emitted_transition_must_be_nonempty': True,
    'full_output_prefix_decode_must_equal_guarded_text': True,
    'scope': 'Narrower token vocabulary; valid split-codepoint tokens and literal U+FFFD are excluded',
}


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class StandaloneTokenPolicy:
    eligible_token_ids: frozenset
    eos_token_ids: frozenset
    decoder: object


def build_token_eligibility(tokenizer, data):
    """Freeze an independently reconstructable inventory; never strip U+FFFD."""
    def decoder(tokens):
        return tokenizer.decode(list(tokens), skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    eos = data.eos_token_id if isinstance(data.eos_token_id, list) else [data.eos_token_id]
    regular = sorted(data.tokenizer_tree.tokens_to_strs)
    expected_regular = set(range(len(tokenizer))) - set(tokenizer.all_special_ids)
    if set(regular) != expected_regular:
        raise ValueError('Regular-token inventory differs from complete tokenizer vocabulary')
    if set(eos).intersection(regular):
        raise ValueError('EOS must not appear in regular-token inventory')
    eligible, excluded, decode_hash = [], {}, hashlib.sha256()
    for token in regular:
        decoded = decoder([token])
        decode_hash.update(canonical([token, decoded]) + b'\n')
        if not decoded:
            excluded[str(token)] = 'empty_standalone_decode'
        elif '\ufffd' in decoded:
            excluded[str(token)] = 'replacement_character_in_standalone_decode'
        else:
            eligible.append(token)
    artifact = {'schema': 'third-token-eligibility-v5', 'policy': TOKEN_POLICY,
        'policy_sha256': hashlib.sha256(canonical(TOKEN_POLICY)).hexdigest(),
        'eligible_token_ids': eligible, 'excluded_token_reasons': excluded,
        'eos_token_ids': sorted(eos), 'regular_token_count': len(regular),
        'tokenizer_length': len(tokenizer), 'special_token_ids': sorted(tokenizer.all_special_ids),
        'eligible_token_count': len(eligible), 'excluded_token_count': len(excluded),
        'standalone_decode_stream_sha256': decode_hash.hexdigest(),
        'standalone_decode_stream_format': 'Sorted regular IDs; canonical JSON [token_id,decoded_text] plus LF'}
    return StandaloneTokenPolicy(frozenset(eligible), frozenset(eos), decoder), artifact
WHITESPACE_HEX = tuple(f'{value:04x}' for value in range(0x10000) if chr(value).isspace())
ESCAPES = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}


@dataclass(frozen=True)
class LexicalState:
    containers: tuple = ()
    expecting_key: bool = False
    pending_key: str = ''
    role: str = ''  # empty means outside a string; otherwise key/reason/other
    key: str = ''
    escape: bool = False
    unicode_digits: str | None = None
    words: int = 0
    inside_word: bool = False

    def accept_decoded(self, character):
        if self.role == 'key':
            return replace(self, key=self.key + character)
        if self.role != 'reason':
            return self
        space = character.isspace()
        count = self.words + int(not space and not self.inside_word)
        if count > WORD_LIMIT:
            raise ValueError('Decoded reason would start word81')
        return replace(self, words=count, inside_word=not space)

    def step(self, character):
        if not self.role:
            if character == '"':
                role = 'key' if self.expecting_key else ('reason' if self.pending_key == 'reason' and self.containers == ('{',) else 'other')
                return replace(self, role=role, key='', escape=False, unicode_digits=None)
            if character in '{[':
                return replace(self, containers=self.containers + (character,), expecting_key=character == '{')
            if character in '}]':
                return replace(self, containers=self.containers[:-1], expecting_key=False)
            if character == ':' and self.containers == ('{',):
                return replace(self, expecting_key=False)
            if character == ',' and self.containers == ('{',):
                return replace(self, expecting_key=True, pending_key='')
            return self
        at_word_boundary_limit = self.role == 'reason' and self.words == WORD_LIMIT and not self.inside_word
        if self.unicode_digits is not None:
            digits = self.unicode_digits + character
            if character.lower() not in '0123456789abcdef':
                raise ValueError('Invalid Unicode escape digit')
            # Avoid entering an escape prefix with no legal completion. Testing
            # only the final nibble could strand a decoder inside e.g. \\uFFF.
            if at_word_boundary_limit and not any(code.startswith(digits.lower()) for code in WHITESPACE_HEX):
                raise ValueError('Unicode escape cannot finish as whitespace')
            if len(digits) == 4:
                return replace(self, unicode_digits=None).accept_decoded(chr(int(digits, 16)))
            return replace(self, unicode_digits=digits)
        if self.escape:
            if character == 'u':
                return replace(self, escape=False, unicode_digits='')
            if character not in ESCAPES:
                raise ValueError('Invalid JSON escape')
            return replace(self, escape=False).accept_decoded(ESCAPES[character])
        if character == '\\':
            return replace(self, escape=True)
        if character == '"':
            return replace(self, role='', pending_key=self.key if self.role == 'key' else self.pending_key)
        return self.accept_decoded(character)


def exact_config(alphabet):
    return CharacterLevelParserConfig(alphabet=alphabet, max_consecutive_whitespaces=12,
                                     force_json_field_order=False, max_json_array_length=8)


class BoundedReasonParser(CharacterLevelParser):
    def __init__(self, inner, config, lexical=None):
        super().__init__(config)
        self.inner = inner
        self.lexical = lexical or LexicalState()
        self._allowed = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, proposed):
        # TokenEnforcer 0.11.3 overwrites root config during initialization.
        # The tokenizer alphabet is known before construction; accept only its
        # identical value and retain explicitly pinned non-alphabet settings.
        if proposed.alphabet != self._config.alphabet:
            raise ValueError('TokenEnforcer attempted to change pinned alphabet')

    def get_allowed_characters(self):
        if self._allowed is None:
            allowed = self.inner.get_allowed_characters()
            state = self.lexical
            if state.role != 'reason' or state.words < WORD_LIMIT or state.inside_word:
                # Until a new decoded word would exceed the bound, the schema's
                # character set is unchanged. Do not scan the whole tokenizer
                # alphabet at every vocabulary-trie node.
                self._allowed = allowed
            else:
                if state.unicode_digits is not None:
                    prefix = state.unicode_digits.lower()
                    candidates = ''.join(sorted({code[len(prefix)] for code in WHITESPACE_HEX if code.startswith(prefix)}))
                    candidates += candidates.upper()
                elif state.escape:
                    candidates = 'unrtf'
                else:
                    candidates = ''.join(chr(int(code, 16)) for code in WHITESPACE_HEX) + '\\"'
                self._allowed = ''.join(c for c in candidates if c in allowed)
        return self._allowed

    def add_character(self, character):
        if character not in self.get_allowed_characters():
            raise ValueError('Character violates schema or decoded reason bound')
        return BoundedReasonParser(self.inner.add_character(character), self._config, self.lexical.step(character))

    def can_end(self):
        return not self.lexical.role and self.inner.can_end()

    def shortcut_key(self):
        return None  # LMFE json_freetext shortcut would bypass this guard.

    def cache_key(self):
        return None  # No cache key may omit lexical word/escape state.


def make_parser(schema, alphabet):
    config = exact_config(alphabet)
    return BoundedReasonParser(JsonSchemaParser(schema, config=config), config)


def make_token_enforcer(schema, tokenizer_data):
    from lmformatenforcer import TokenEnforcer
    parser = make_parser(schema, tokenizer_data.tokenizer_alphabet)
    enforcer = TokenEnforcer(tokenizer_data, parser)
    assert enforcer.root_parser.config == exact_config(tokenizer_data.tokenizer_alphabet)
    return enforcer


def replay(raw, schema, alphabet):
    parser = make_parser(schema, alphabet)
    for character in raw:
        parser = parser.add_character(character)
    if not parser.can_end():
        raise ValueError('Incomplete JSON/word-guard state')
    return parser


# Greedy implementation of the prospectively pinned proposal-plus-guard policy.
from dataclasses import dataclass
import time
import math
from lmformatenforcer import TokenEnforcer, JsonSchemaParser


def make_base_enforcer(schema, data):
    config = exact_config(data.tokenizer_alphabet)
    parser = JsonSchemaParser(schema, config=config)
    enforcer = TokenEnforcer(data, parser)
    # LMFE changes the entire config during initialization; restore every pin.
    # The parser context alphabet was already initialized correctly above.
    parser.config = config
    assert enforcer.root_parser.config == config
    return enforcer


@dataclass(frozen=True)
class GuardTokenState:
    parser: object
    current_word_tokens: tuple = ()
    ended: bool = False
    output_tokens: tuple = ()
    emitted_text: str = ''

    def transition(self, token, data, policy):
        if self.ended:
            raise ValueError('Token after EOS')
        if token in policy.eos_token_ids:
            if not self.parser.can_end():
                raise ValueError('Premature EOS')
            if policy.decoder(self.output_tokens) != self.emitted_text:
                raise ValueError('Full output prefix decode differs from guarded text')
            return replace(self, ended=True)
        if token not in policy.eligible_token_ids:
            raise ValueError('Token violates standalone-valid UTF-8 policy')
        tree = data.tokenizer_tree
        if token not in tree.tokens_to_strs:
            raise ValueError('Unknown regular token')
        if token in tree.new_word_tokens:
            new_word = (token,)
            new_characters = tree.tokens_to_strs[token]
        else:
            new_word = self.current_word_tokens + (token,)
            prior = data.decoder(list(self.current_word_tokens))
            decoded = data.decoder(list(new_word))
            if not decoded.startswith(prior):
                raise ValueError('Non-prefix-stable tokenizer transition')
            new_characters = decoded[len(prior):]
        if not new_characters:
            raise ValueError('Non-EOS token emitted no guarded characters')
        output_tokens = self.output_tokens + (token,)
        emitted_text = self.emitted_text + new_characters
        if policy.decoder(output_tokens) != emitted_text:
            raise ValueError('Full output prefix decode differs from guarded text')
        parser = self.parser
        for character in new_characters:
            parser = parser.add_character(character)
        return GuardTokenState(parser, new_word, False, output_tokens, emitted_text)


class CandidateExhaustedError(ValueError):
    def __init__(self, candidates, rejected):
        super().__init__('No finite native-proposal token satisfies the exact guard')
        self.candidates = candidates
        self.rejected = rejected


def choose_ranked(ranked_tokens, state, data, policy):
    rejected = {}
    count = 0
    for token in ranked_tokens:
        count += 1
        try:
            next_state = state.transition(int(token), data, policy)
        except ValueError as error:
            key = str(error)
            rejected[key] = rejected.get(key, 0) + 1
            continue
        return int(token), next_state, count, rejected
    raise CandidateExhaustedError(count, rejected)


def rank_finite_scores(scores):
    """Descending finite score, then lowest token ID; -inf is the base mask."""
    values = list(scores)
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError('NaN or positive infinity in generation scores')
    return sorted((index for index, value in enumerate(values) if math.isfinite(value)),
                  key=lambda index: (-values[index], index))


class GreedyGuardProcessor:
    def __init__(self, schema, data, base_enforcer, input_ids, policy):
        self.data = data
        self.policy = policy
        self.base_enforcer = base_enforcer
        self.expected_input_ids = list(input_ids)
        self.state = GuardTokenState(make_parser(schema, data.tokenizer_alphabet))
        self.steps = []

    def __call__(self, input_ids, scores):
        import torch
        started = time.perf_counter()
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError('Greedy guard supports exactly one beam and one row')
        actual = input_ids[0].tolist()
        if actual != self.expected_input_ids:
            raise ValueError('Generation departed from the previously selected token')
        ranked = rank_finite_scores(scores[0].tolist())
        proposal = set(self.base_enforcer.get_allowed_tokens(actual).allowed_tokens)
        if any(token not in proposal for token in ranked):
            raise ValueError('Native prefix mask did not precede custom guard')
        try:
            selected, new_state, candidates, rejected = choose_ranked(ranked, self.state, self.data, self.policy)
        except CandidateExhaustedError as error:
            self.steps.append({'step': len(self.steps), 'accepted_token_id': None,
                'finite_native_candidates': len(ranked), 'candidates_considered': error.candidates,
                'rejected_candidates': error.rejected, 'prefix_mask_verified': True,
                'selection_seconds': time.perf_counter() - started})
            raise
        result = torch.full_like(scores, float('-inf'))
        result[0, selected] = scores[0, selected]
        self.state = new_state
        self.expected_input_ids.append(selected)
        self.steps.append({'step': len(self.steps), 'accepted_token_id': selected,
            'finite_native_candidates': len(ranked), 'candidates_considered': candidates,
            'rejected_candidates': rejected, 'prefix_mask_verified': True,
            'selection_seconds': time.perf_counter() - started})
        return result


def replay_guard_tokens(input_ids, output_ids, schema, data, policy):
    base = make_base_enforcer(schema, data)
    state = GuardTokenState(make_parser(schema, data.tokenizer_alphabet))
    sequence = list(input_ids)
    for token in output_ids:
        if token not in base.get_allowed_tokens(sequence).allowed_tokens:
            raise ValueError('Generated token violates native proposal filter')
        state = state.transition(token, data, policy)
        sequence.append(token)
    return state
