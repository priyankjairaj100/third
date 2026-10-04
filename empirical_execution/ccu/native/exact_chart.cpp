// Exact rational factor canonicalizer. No original data access.
// Uses installed GMP rational API. Public ABI declarations for LP64 Linux:
// https://gmplib.org/manual/Integer-Internals
// https://gmplib.org/manual/Rational-Internals
// https://gmplib.org/manual/Initializing-Rationals
// Build: g++ -std=c++17 -O2 exact_chart.cpp -Wl,-l:libgmp.so.10 -o exact_chart
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <utility>
extern "C" {
struct Z { int alloc; int size; unsigned long* limbs; };
struct Qraw { Z num; Z den; };
void __gmpq_init(Qraw*); void __gmpq_clear(Qraw*);
void __gmpq_set(Qraw*, const Qraw*);
void __gmpq_set_si(Qraw*, long, unsigned long);
int __gmpq_set_str(Qraw*,const char*,int);
void __gmpq_canonicalize(Qraw*);
void __gmpq_add(Qraw*,const Qraw*,const Qraw*);
void __gmpq_sub(Qraw*,const Qraw*,const Qraw*);
void __gmpq_mul(Qraw*,const Qraw*,const Qraw*);
void __gmpq_div(Qraw*,const Qraw*,const Qraw*);
int __gmpq_equal(const Qraw*,const Qraw*);
char* __gmpq_get_str(char*,int,const Qraw*);
void __gmp_get_memory_functions(void *(**)(size_t),void *(**)(void*,size_t,size_t),void (**)(void*,size_t));
extern const char* __gmp_version;
}
static_assert(sizeof(int)==4 && sizeof(unsigned long)==8 && sizeof(void*)==8,"LP64 GMP ABI required");
class Q {
public:
 Qraw q;
 Q(){__gmpq_init(&q);} Q(long n){__gmpq_init(&q);__gmpq_set_si(&q,n,1);}
 Q(const Q&o){__gmpq_init(&q);__gmpq_set(&q,&o.q);}
 Q(Q&&o) noexcept {__gmpq_init(&q);std::swap(q,o.q);}
 ~Q(){__gmpq_clear(&q);}
 Q& operator=(const Q&o){if(this!=&o)__gmpq_set(&q,&o.q);return *this;}
 Q& operator=(Q&&o) noexcept {std::swap(q,o.q);return *this;}
 bool zero()const{return q.num.size==0;}
 void parse(const std::string&s){if(__gmpq_set_str(&q,s.c_str(),10))throw std::runtime_error("bad rational");__gmpq_canonicalize(&q);}
 std::string str()const {char*p=__gmpq_get_str(nullptr,10,&q);std::string s(p);void(*f)(void*,size_t);__gmp_get_memory_functions(nullptr,nullptr,&f);f(p,s.size()+1);return s;}
 friend Q operator+(const Q&a,const Q&b){Q z;__gmpq_add(&z.q,&a.q,&b.q);return z;}
 friend Q operator-(const Q&a,const Q&b){Q z;__gmpq_sub(&z.q,&a.q,&b.q);return z;}
 friend Q operator*(const Q&a,const Q&b){Q z;__gmpq_mul(&z.q,&a.q,&b.q);return z;}
 friend Q operator/(const Q&a,const Q&b){if(b.zero())throw std::runtime_error("zero division");Q z;__gmpq_div(&z.q,&a.q,&b.q);return z;}
};
using M=std::vector<std::vector<Q>>;
M zeros(int a,int b){return M(a,std::vector<Q>(b));}
M tr(const M&a,int cols){M b=zeros(cols,a.size());for(size_t i=0;i<a.size();++i)for(int j=0;j<cols;++j)b[j][i]=a[i][j];return b;}
M mul(const M&a,const M&b,int outcols){M z=zeros(a.size(),outcols);for(size_t i=0;i<a.size();++i)for(size_t k=0;k<b.size();++k)if(!a[i][k].zero())for(int j=0;j<outcols;++j)if(!b[k][j].zero())z[i][j]=z[i][j]+a[i][k]*b[k][j];return z;}
std::pair<M,std::vector<int>> rref(M a,int cols){
 std::vector<int> piv;size_t rr=0;
 for(int j=0;j<cols && rr<a.size();++j){size_t p=rr;while(p<a.size()&&a[p][j].zero())++p;if(p==a.size())continue;std::swap(a[rr],a[p]);Q div=a[rr][j];for(int k=j;k<cols;++k)a[rr][k]=a[rr][k]/div;
 for(size_t i=0;i<a.size();++i)if(i!=rr&&!a[i][j].zero()){Q f=a[i][j];a[i][j]=Q(0);for(int k=j+1;k<cols;++k)if(!a[rr][k].zero())a[i][k]=a[i][k]-f*a[rr][k];}piv.push_back(j);++rr;
 }a.resize(rr);return {a,piv};
}
M read(int a,int b){M m=zeros(a,b);std::string x;for(auto&r:m)for(auto&v:r){if(!(std::cin>>x))throw std::runtime_error("short input");v.parse(x);}return m;}
void write(const M&m){for(const auto&r:m){for(const auto&q:r)std::cout<<q.str()<<' ';std::cout<<'\n';}}
int main(int argc,char**argv){try{
 Q a,b;a.parse("2/6");b.parse("7/11");if((a+b).str()!="32/33"||(a*b).str()!="7/33"||(a/b).str()!="11/21")throw std::runtime_error("GMP arithmetic self-test failed");
 if(argc>1&&std::string(argv[1])=="--self-test"){std::cout<<"GMP "<<__gmp_version<<" exact rational ABI and arithmetic checks passed\n";return 0;}
 int d,t,c;long long count;if(!(std::cin>>d>>t>>c>>count)||d<1||t<0||c<1)throw std::runtime_error("bad dimensions");
 M w=read(d,t),a0=read(t,t),b0=read(t,c);
 auto first=rref(tr(w,t),d);M u=tr(first.first,d);int r=first.second.size();M coord=zeros(r,t);for(int i=0;i<r;++i)coord[i]=w[first.second[i]];
 M aa=mul(mul(coord,a0,t),tr(coord,t),r),bb=mul(coord,b0,c);
 M joint=aa;for(int i=0;i<r;++i)joint[i].insert(joint[i].end(),bb[i].begin(),bb[i].end());
 auto second=rref(tr(joint,r+c),r);int s=second.second.size();
 std::vector<int> piv=first.second;
 if(s<r){M lower=tr(second.first,r);M z=mul(u,lower,s);auto last=rref(tr(z,s),d);M rr=zeros(s,r);for(int i=0;i<s;++i)rr[i]=u[last.second[i]];aa=mul(mul(rr,aa,r),tr(rr,r),s);bb=mul(rr,bb,c);u=tr(last.first,d);piv=last.second;}
 std::cout<<d<<' '<<s<<' '<<c<<' '<<count<<'\n';for(int p:piv)std::cout<<p<<' ';std::cout<<'\n';write(u);write(aa);write(bb);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
