// Vector math, dimensionality reduction and clustering. No DOM, no dependencies.

export function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
export function gaussFn(rnd){return function(){let u=0,v=0;while(u===0)u=rnd();while(v===0)v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};}
export function fnv1a(str){let h=0x811c9dc5;for(let i=0;i<str.length;i++){h^=str.charCodeAt(i);h=Math.imul(h,0x01000193);}return h>>>0;}
export function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
export function vnorm(a){return Math.sqrt(dot(a,a));}
export function normalizeInPlace(v){const s=vnorm(v);if(s>0)for(let i=0;i<v.length;i++)v[i]/=s;return v;}
export function cosineSim(a,b){const d=vnorm(a)*vnorm(b);return d?dot(a,b)/d:0;}
export function euclid(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d;}return Math.sqrt(s);}


// ---------- PCA by power iteration (works for wide data, cost n*d per step) ----------
export function pca(vectors,nComp,{exclude=[],seed=7,maxIter=60}={}){
  const n=vectors.length,d=vectors[0].length;
  const mean=new Float64Array(d);
  for(const v of vectors)for(let i=0;i<d;i++)mean[i]+=v[i];
  for(let i=0;i<d;i++)mean[i]/=n;
  const X=vectors.map(v=>{const c=new Float64Array(d);for(let i=0;i<d;i++)c[i]=v[i]-mean[i];return c;});
  for(const e of exclude)for(const x of X){const p=dot(x,e);for(let i=0;i<d;i++)x[i]-=p*e[i];}
  let total=0;for(const x of X)total+=dot(x,x);
  const rnd=mulberry32(seed);const comps=[],varRatio=[];
  const orth=(w)=>{for(const c of comps.concat(exclude)){const p=dot(w,c);for(let i=0;i<d;i++)w[i]-=p*c[i];}};
  for(let c=0;c<nComp;c++){
    let v=new Float64Array(d);for(let i=0;i<d;i++)v[i]=rnd()-0.5;orth(v);normalizeInPlace(v);
    let lambda=0;
    for(let it=0;it<maxIter;it++){
      const w=new Float64Array(d);
      for(const x of X){const p=dot(x,v);if(p!==0)for(let i=0;i<d;i++)w[i]+=p*x[i];}
      orth(w);const nw=vnorm(w);if(nw<1e-12)break;
      for(let i=0;i<d;i++)w[i]/=nw;
      const diff=1-Math.abs(dot(w,v));v=w;lambda=nw;if(diff<1e-7)break;
    }
    comps.push(v);varRatio.push(total>0?lambda/total:0);
  }
  const coords=X.map(x=>comps.map(c=>dot(x,c)));
  return {mean,comps,coords,varRatio};
}

// Random Gaussian projection to keep t-SNE and UMAP fast on wide vectors
export function reduceForLayout(vectors,target=96,seed=11){
  const d=vectors[0].length;
  const unit=vectors.map(v=>normalizeInPlace(Float64Array.from(v)));
  if(d<=target)return unit.map(v=>Array.from(v));
  const g=gaussFn(mulberry32(seed));const R=[];
  for(let k=0;k<target;k++){const r=new Float64Array(d);for(let i=0;i<d;i++)r[i]=g()/Math.sqrt(target);R.push(r);}
  return unit.map(v=>R.map(r=>dot(v,r)));
}

// ---------- t-SNE (exact, in the style of tsnejs) ----------
export class TSNE{
  constructor(X,{dim=2,perplexity=30,epsilon=10,seed=3}={}){
    const n=X.length;this.n=n;this.dim=dim;this.epsilon=epsilon;this.iter=0;
    const D=new Float64Array(n*n);
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){let s=0;const a=X[i],b=X[j];for(let k=0;k<a.length;k++){const q=a[k]-b[k];s+=q*q;}D[i*n+j]=s;D[j*n+i]=s;}
    this.P=TSNE.d2p(D,n,Math.max(2,Math.min(perplexity,(n-1)/3)),1e-4);
    const g=gaussFn(mulberry32(seed));
    this.Y=new Float64Array(n*dim);for(let i=0;i<n*dim;i++)this.Y[i]=g()*1e-4;
    this.gains=new Float64Array(n*dim).fill(1);this.ystep=new Float64Array(n*dim);
    this.num=new Float64Array(n*n);this.grad=new Float64Array(n*dim);
  }
  static d2p(D,n,perplexity,tol){
    const Htarget=Math.log(perplexity);const P=new Float64Array(n*n);const prow=new Float64Array(n);
    for(let i=0;i<n;i++){
      let betamin=-Infinity,betamax=Infinity,beta=1;
      for(let num=0;num<50;num++){
        let psum=0;
        for(let j=0;j<n;j++){const pj=j===i?0:Math.exp(-D[i*n+j]*beta);prow[j]=pj;psum+=pj;}
        let H=0;
        for(let j=0;j<n;j++){const pj=psum===0?0:prow[j]/psum;prow[j]=pj;if(pj>1e-7)H-=pj*Math.log(pj);}
        if(H>Htarget){betamin=beta;beta=betamax===Infinity?beta*2:(beta+betamax)/2;}
        else{betamax=beta;beta=betamin===-Infinity?beta/2:(beta+betamin)/2;}
        if(Math.abs(H-Htarget)<tol)break;
      }
      for(let j=0;j<n;j++)P[i*n+j]=prow[j];
    }
    const Pout=new Float64Array(n*n);const N2=n*2;
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)Pout[i*n+j]=Math.max((P[i*n+j]+P[j*n+i])/N2,1e-100);
    return Pout;
  }
  step(){
    const n=this.n,dim=this.dim,Y=this.Y,P=this.P,num=this.num,grad=this.grad;
    const exag=this.iter<100?4:1;
    let qsum=0;
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
      let s=0;for(let d=0;d<dim;d++){const q=Y[i*dim+d]-Y[j*dim+d];s+=q*q;}
      const q=1/(1+s);num[i*n+j]=q;num[j*n+i]=q;qsum+=2*q;
    }
    grad.fill(0);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++){
      if(i===j)continue;const q=num[i*n+j];
      const pre=4*(exag*P[i*n+j]-q/qsum)*q;
      for(let d=0;d<dim;d++)grad[i*dim+d]+=pre*(Y[i*dim+d]-Y[j*dim+d]);
    }
    const mom=this.iter<250?0.5:0.8;const ymean=new Float64Array(dim);
    for(let k=0;k<n*dim;k++){
      const gid=grad[k],sid=this.ystep[k];
      let gain=Math.sign(gid)===Math.sign(sid)?this.gains[k]*0.8:this.gains[k]+0.2;
      if(gain<0.01)gain=0.01;this.gains[k]=gain;
      const ns=mom*sid-this.epsilon*gain*gid;this.ystep[k]=ns;Y[k]+=ns;ymean[k%dim]+=Y[k];
    }
    for(let k=0;k<n*dim;k++)Y[k]-=ymean[k%dim]/n;
    this.iter++;
  }
  coords(){const out=[];for(let i=0;i<this.n;i++){const r=[];for(let d=0;d<this.dim;d++)r.push(this.Y[i*this.dim+d]);out.push(r);}return out;}
}

// ---------- k-means (cosine, k-means++ seeding) ----------
export function kmeans(vectors,k,seed=5){
  const n=vectors.length;k=Math.max(1,Math.min(k,n));
  const X=vectors.map(v=>normalizeInPlace(Float64Array.from(v)));const rnd=mulberry32(seed);const d=X[0].length;
  const C=[Float64Array.from(X[Math.floor(rnd()*n)])];
  const dist=new Float64Array(n).fill(Infinity);
  while(C.length<k){
    let sum=0;const last=C[C.length-1];
    for(let i=0;i<n;i++){const di=Math.max(0,1-dot(X[i],last));if(di<dist[i])dist[i]=di;sum+=dist[i];}
    let r=rnd()*sum,pick=0;for(let i=0;i<n;i++){r-=dist[i];if(r<=0){pick=i;break;}}
    C.push(Float64Array.from(X[pick]));
  }
  const assign=new Int32Array(n);
  for(let it=0;it<30;it++){
    let changed=0;
    for(let i=0;i<n;i++){let best=0,bs=-Infinity;for(let c=0;c<k;c++){const s=dot(X[i],C[c]);if(s>bs){bs=s;best=c;}}if(assign[i]!==best||it===0){changed++;assign[i]=best;}}
    const sums=Array.from({length:k},()=>new Float64Array(d));const cnt=new Int32Array(k);
    for(let i=0;i<n;i++){const s=sums[assign[i]];for(let j=0;j<d;j++)s[j]+=X[i][j];cnt[assign[i]]++;}
    for(let c=0;c<k;c++)if(cnt[c]>0)C[c]=normalizeInPlace(sums[c]);
    if(changed===0)break;
  }
  return Array.from(assign);
}

// Scale coordinates so the cloud fits a unit sphere
export function fitUnit(coords,dim){
  const n=coords.length;const out=new Float32Array(n*3);if(!n)return out;
  const mean=[0,0,0];for(const c of coords)for(let d=0;d<dim;d++)mean[d]+=c[d]/n;
  let r=0;for(const c of coords){let s=0;for(let d=0;d<dim;d++){const q=c[d]-mean[d];s+=q*q;}if(s>r)r=s;}
  r=Math.sqrt(r)||1;
  coords.forEach((c,i)=>{for(let d=0;d<dim;d++)out[i*3+d]=(c[d]-mean[d])/r;});
  return {xyz:out,mean,scale:r};
}

