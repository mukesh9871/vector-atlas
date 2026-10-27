// Vector Atlas: UI state, projections, canvas renderer and interaction.
import {mulberry32,dot,vnorm,normalizeInPlace,euclid,pca,reduceForLayout,TSNE,kmeans,fitUnit} from "./math.js";
import {LexicalEmbedder,NeuralEmbedder,MODELS,loadModel} from "./embedders.js";
import {parseVectors,parseMeta} from "./parsers.js";
import {buildDemo,SAMPLE_TOPICS,SAMPLE_REVIEWS} from "./samples.js";

const $=(id)=>document.getElementById(id);
const fmt=(n)=>Number(n).toLocaleString("en-US");
const reduceMotion=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const PALETTE_LIGHT=["#2C43C8","#D2552D","#1E9A6B","#B0359B","#B98700","#2A8FB8","#7A4FD6","#C2364A","#5E8A1E","#8A5A2B","#3B6E8F","#D46A9C"];
const PALETTE_DARK=["#8E9DFF","#FF8A5C","#47D19A","#E07AD3","#F2C14E","#5CC8F0","#B496FF","#FF6F85","#A3D660","#D9A36B","#7FB3D5","#FFA6CC"];
const CAM=3.2;
const FONT='"Schibsted Grotesk", ui-sans-serif, system-ui, sans-serif';
const DEFAULT_VIEW={yaw:0.7,pitch:-0.38,zoom:1,panX:0,panY:0};

const S={
  items:[],dim:0,source:null,embedder:null,norms:null,meta:null,
  method:"pca",ndPref:3,xyz:new Float32Array(0),fit:null,pcaInfo:null,customInfo:null,axisLabels:null,
  layout:null,paused:false,
  colorBy:"group",k:6,clusters:null,groups:[],groupIndex:new Map(),off:new Set(),
  psize:5,allLabels:false,autoRotate:!reduceMotion,
  selected:-1,neighbors:[],hover:-1,isolate:false,matches:null,query:null,
  view:{...DEFAULT_VIEW},dirty:true,drag:null,
  scr:new Float32Array(0),vis:new Uint8Array(0),order:new Int32Array(0),
};
let colors={};

function effNd(){return S.method==="custom"?2:S.ndPref;}
function isDark(){const t=document.documentElement.getAttribute("data-theme");if(t)return t==="dark";return window.matchMedia("(prefers-color-scheme: dark)").matches;}
function readColors(){
  const cs=getComputedStyle(document.documentElement);const g=(n)=>cs.getPropertyValue(n).trim();
  colors={ink:g("--ink"),muted:g("--muted"),line:g("--line"),accent:g("--accent"),stage:g("--stage"),panel:g("--panel"),palette:isDark()?PALETTE_DARK:PALETTE_LIGHT};
  $("themeBtn").textContent=isDark()?"Light theme":"Dark theme";
  S.dirty=true;renderLegend();
}

function errText(e){
  if(e&&e.name==="AbortError")return "Stopped.";
  return String((e&&e.message)||e).replace(/\.$/,"");
}

// ---------- data ----------
function setData(items,{source,embedder,dim,meta=null}){
  stopLayout();
  S.items=items;S.source=source;S.embedder=embedder;S.dim=dim;S.meta=meta;
  S.norms=new Float32Array(items.map(it=>vnorm(it.vec)));
  S.selected=-1;S.neighbors=[];S.query=null;S.hover=-1;S.isolate=false;S.off=new Set();S.clusters=null;
  S.view={...DEFAULT_VIEW};
  computeGroups();
  if(S.colorBy==="cluster"&&!S.clusters)computeClusters();
  renderLegend();
  $("queryList").innerHTML="";
  runProjection();
  updateSummary();updateQueryUI();renderSelection();updateSearch();updateCustomHint();
  $("empty").hidden=items.length>0;
}
function computeGroups(){
  const counts=new Map();
  for(const it of S.items){if(it.group!=null&&it.group!=="")counts.set(it.group,(counts.get(it.group)||0)+1);}
  S.groups=[...counts.entries()].sort((a,b)=>b[1]-a[1]);
  S.groupIndex=new Map(S.groups.map((g,i)=>[g[0],Math.min(i,11)]));
  if(S.colorBy==="group"&&!S.groups.length&&S.items.length>1){$("colorBy").value="cluster";S.colorBy="cluster";S.autoCluster=true;$("kWrap").hidden=false;}
  else if(S.autoCluster&&S.colorBy==="cluster"&&S.groups.length){$("colorBy").value="group";S.colorBy="group";S.autoCluster=false;$("kWrap").hidden=true;}
  if(S.colorBy==="cluster"&&!S.clusters)computeClusters();
  renderLegend();S.dirty=true;
}
function computeClusters(){
  if(!S.items.length){S.clusters=null;return;}
  S.clusters=kmeans(S.items.map(it=>it.vec),Math.min(S.k,S.items.length));
}
function pointColor(i){
  const pal=colors.palette;
  if(S.colorBy==="cluster"&&S.clusters)return pal[S.clusters[i]%pal.length];
  if(S.colorBy==="group"){const g=S.items[i].group;const gi=S.groupIndex.get(g);if(gi===undefined)return colors.muted;if(gi>=11&&S.groups.length>12)return colors.muted;return pal[gi];}
  return colors.accent;
}
function legendKey(i){
  if(S.colorBy==="cluster"&&S.clusters)return "c"+S.clusters[i];
  if(S.colorBy==="group"){const gi=S.groupIndex.get(S.items[i].group);if(gi===undefined)return "g:none";if(gi>=11&&S.groups.length>12)return "g:other";return "g:"+S.items[i].group;}
  return null;
}
function renderLegend(){
  const el=$("legend");el.innerHTML="";
  if(!S.items.length||!colors.palette)return;
  const entries=[];
  if(S.colorBy==="cluster"&&S.clusters){
    const cnt=new Map();S.clusters.forEach(c=>cnt.set(c,(cnt.get(c)||0)+1));
    [...cnt.keys()].sort((a,b)=>a-b).forEach(c=>entries.push({key:"c"+c,name:"Cluster "+(c+1),color:colors.palette[c%12],n:cnt.get(c)}));
  }else if(S.colorBy==="group"&&S.groups.length){
    const many=S.groups.length>12;
    S.groups.slice(0,many?11:12).forEach(([g,n],i)=>entries.push({key:"g:"+g,name:g,color:colors.palette[i],n}));
    if(many)entries.push({key:"g:other",name:"Other groups",color:colors.muted,n:S.groups.slice(11).reduce((s,g)=>s+g[1],0)});
    const none=S.items.length-S.groups.reduce((s,g)=>s+g[1],0);
    if(none>0)entries.push({key:"g:none",name:"No group",color:colors.muted,n:none});
  }
  for(const e of entries){
    const b=document.createElement("button");b.type="button";if(S.off.has(e.key))b.className="off";
    b.setAttribute("aria-pressed",S.off.has(e.key)?"false":"true");b.title="Click to fade or restore this set";
    const i=document.createElement("i");i.style.background=e.color;
    const s=document.createElement("span");s.textContent=e.name;
    const m=document.createElement("em");m.textContent=fmt(e.n);
    b.append(i,s,m);
    b.onclick=()=>{S.off.has(e.key)?S.off.delete(e.key):S.off.add(e.key);renderLegend();S.dirty=true;};
    el.append(b);
  }
}
function updateSummary(){
  const n=S.items.length;
  const src={demo:"concept words with synthetic demo vectors",lexical:"your text, embedded by words and subwords",neural:"your text, embedded by "+(S.embedder&&S.embedder.model?S.embedder.model.name:"a neural model"),imported:"imported vectors"}[S.source]||"";
  $("summary").textContent=n?fmt(n)+" points, "+fmt(S.dim)+" dimensions, "+src:"No data yet";
}

// ---------- projections ----------
function stopLayout(){S.layout=null;$("progress").hidden=true;}
function setProgress(t){$("progressText").textContent=t;$("progress").hidden=!t;}
function showOpts(){
  for(const m of ["pca","tsne","umap","custom"])$("opt-"+m).hidden=S.method!==m;
  $("rerunRow").hidden=!(S.method==="tsne"||S.method==="umap");
  document.querySelectorAll("#methodSeg button").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.m===S.method)));
  document.querySelectorAll("#dimSeg button").forEach(b=>{b.setAttribute("aria-pressed",String(Number(b.dataset.d)===effNd()));b.disabled=S.method==="custom"&&b.dataset.d==="3";});
  updateHud();
}
function applyCoords(coords,nd){
  const f=fitUnit(coords,nd);S.xyz=f.xyz;S.fit=f;S.dirty=true;
}
function runProjection(){
  stopLayout();S.pcaInfo=null;S.customInfo=null;S.axisLabels=null;
  showOpts();
  const n=S.items.length;const nd=effNd();
  $("pcaVar").textContent="";
  if(!n){S.xyz=new Float32Array(0);S.dirty=true;return;}
  const vecs=S.items.map(it=>it.vec);
  if(S.method==="pca"||S.method==="custom"&&!S.customInfo){
    if(S.method==="custom"){if(!customReady()){runPcaInto(vecs,2);$("customStatus").textContent="Showing PCA until you apply axes.";$("customStatus").className="status";return;}applyCustom();return;}
    runPcaInto(vecs,nd);return;
  }
  if(S.method==="tsne"){
    if(n<6){setHud("t-SNE needs at least 6 points.");runPcaInto(vecs,nd);return;}
    if(n>1500){setHud("t-SNE runs on up to 1,500 points here. Use PCA or UMAP for this set.");runPcaInto(vecs,nd);return;}
    setProgress("Preparing t-SNE…");
    setTimeout(()=>{
      if(S.method!=="tsne"||S.items.length!==n)return;
      const X=reduceForLayout(vecs);
      const obj=new TSNE(X,{dim:nd,perplexity:Number($("perp").value),epsilon:Number($("lr").value),seed:3});
      S.layout={type:"tsne",obj,total:1000,step:0,nd};S.paused=false;$("pauseBtn").textContent="Pause";
      applyCoords(obj.coords(),nd);
    },30);
    return;
  }
  if(S.method==="umap"){
    const U=window.UMAP&&(window.UMAP.UMAP||window.UMAP);
    if(typeof U!=="function"){setHud("UMAP couldn’t load. Check your connection, or try PCA or t-SNE.");runPcaInto(vecs,nd);return;}
    if(n<4){setHud("UMAP needs at least 4 points.");runPcaInto(vecs,nd);return;}
    setProgress("Preparing UMAP…");
    setTimeout(()=>{
      if(S.method!=="umap"||S.items.length!==n)return;
      try{
        const X=reduceForLayout(vecs);
        const obj=new U({nComponents:nd,nNeighbors:Math.max(2,Math.min(Number($("nn").value),n-1)),minDist:Number($("md").value),random:mulberry32(9)});
        const total=obj.initializeFit(X);
        S.layout={type:"umap",obj,total,step:0,nd};S.paused=false;$("pauseBtn").textContent="Pause";
        applyCoords(obj.getEmbedding(),nd);
      }catch(err){setProgress("");setHud("UMAP couldn’t run on this data: "+err.message);runPcaInto(vecs,nd);}
    },30);
  }
}
function runPcaInto(vecs,nd){
  if(vecs.length<2){applyCoords(vecs.map(()=>[0,0,0]),nd);return;}
  let P;
  if(vecs.length>2000){
    const rnd=mulberry32(21);const idx=vecs.map((_,i)=>i).sort(()=>rnd()-0.5).slice(0,2000);
    P=pca(idx.map(i=>vecs[i]),nd);
    P.coords=vecs.map(v=>P.comps.map(c=>{let s=0;for(let k=0;k<c.length;k++)s+=(v[k]-P.mean[k])*c[k];return s;}));
  }else P=pca(vecs,nd);
  S.pcaInfo=P;applyCoords(P.coords,nd);
  const pct=P.varRatio.map(v=>(v*100).toFixed(1)+"%");
  S.axisLabels=pct.map((p,i)=>["","PC"+(i+1)+", "+p]);
  const tot=(P.varRatio.reduce((a,b)=>a+b,0)*100).toFixed(1);
  $("pcaVar").textContent="These "+nd+" directions explain "+tot+"% of the spread in the original "+fmt(S.dim)+" dimensions.";
}
function stepLayout(){
  const L=S.layout;if(!L)return false;
  if(L.type==="tsne")L.obj.step();else L.obj.step();
  L.step++;
  if(L.step>=L.total){return false;}
  return true;
}
function layoutCoords(){const L=S.layout;return L.type==="tsne"?L.obj.coords():L.obj.getEmbedding();}

// custom axes
function customReady(){return ["xl","xr"].some(id=>$(id).value.trim());}
function updateCustomHint(){
  $("customHint").textContent=S.embedder?"Type a word or phrase for each end. Points slide toward the end whose meaning they resemble most.":"Type a label or a group name for each end, for example “happy” and “sad”, or “animals” and “technology”. Each end is the average of the matching points.";
}
async function axisVec(term){
  if(!term)return null;
  if(S.embedder)return await S.embedder.embed(term);
  const t=term.toLowerCase();
  let idx=S.items.map((it,i)=>[it,i]).filter(([it])=>String(it.label).toLowerCase()===t).map(x=>x[1]);
  if(!idx.length)idx=S.items.map((it,i)=>[it,i]).filter(([it])=>String(it.group||"").toLowerCase()===t).map(x=>x[1]);
  if(!idx.length)idx=S.items.map((it,i)=>[it,i]).filter(([it])=>String(it.label).toLowerCase().includes(t)).map(x=>x[1]);
  if(!idx.length)throw new Error("No point label contains “"+term+"”.");
  const v=new Float64Array(S.dim);for(const i of idx){const w=S.items[i].vec;const nn=S.norms[i]||1;for(let k=0;k<S.dim;k++)v[k]+=w[k]/nn;}
  return normalizeInPlace(v);
}
let customRun=0;
async function applyCustom(){
  const run=++customRun;const st=$("customStatus");
  if(!customReady()){st.textContent="Fill in at least one end of the horizontal axis.";st.className="status err";return;}
  const terms=["xl","xr","yd","yu"].map(id=>$(id).value.trim());
  st.className="status";st.textContent=S.embedder&&S.embedder.kind==="neural"?"Embedding the axis words…":"Building axes…";
  try{
    const [L,R,D,U]=await Promise.all(terms.map(axisVec));
    if(run!==customRun||S.method!=="custom")return;
    const d=S.dim;const m=new Float64Array(d);
    for(let i=0;i<S.items.length;i++){const w=S.items[i].vec;const nn=S.norms[i]||1;for(let k=0;k<d;k++)m[k]+=w[k]/nn/S.items.length;}
    const diff=(a,b)=>{const out=new Float64Array(d);for(let k=0;k<d;k++)out[k]=(a?a[k]:m[k])-(b?b[k]:m[k]);return out;};
    const dirX=diff(R,L);if(vnorm(dirX)<1e-9)throw new Error("Both ends of the horizontal axis point the same way. Try more different words.");
    normalizeInPlace(dirX);
    let dirY=null,P=null;
    if(D||U){dirY=diff(U,D);if(vnorm(dirY)<1e-9)dirY=null;else normalizeInPlace(dirY);}
    const vecs=S.items.map((it,i)=>{const nn=S.norms[i]||1;return Float64Array.from(it.vec,x=>x/nn);});
    if(!dirY)P=pca(vecs,1,{exclude:[dirX]});
    const coords=vecs.map((v,i)=>{let x=0,y=0;for(let k=0;k<d;k++){x+=(v[k]-m[k])*dirX[k];if(dirY)y+=(v[k]-m[k])*dirY[k];}if(P)y=P.coords[i][0];return [x,y];});
    S.customInfo={m,dirX,dirY,P};
    S.axisLabels=[[terms[0]||"average",terms[1]||"average"],dirY?[terms[2]||"average",terms[3]||"average"]:["","main remaining spread"]];
    applyCoords(coords,2);placeQuery();
    st.textContent=S.embedder?"":"Ends are averages of matching labels.";
  }catch(err){
    if(run!==customRun)return;
    st.className="status err";st.textContent=errText(err);
  }
}

// ---------- neighbors and query ----------
function topK(v,k,exclude){
  const cos=$("metric").value==="cosine";const nv=vnorm(v)||1;const out=[];
  for(let j=0;j<S.items.length;j++){
    if(j===exclude)continue;const w=S.items[j].vec;
    const val=cos?dot(v,w)/(nv*(S.norms[j]||1)):euclid(v,w);out.push([j,val]);
  }
  out.sort((a,b)=>cos?b[1]-a[1]:a[1]-b[1]);
  return out.slice(0,k);
}
function select(i){
  S.selected=i;
  S.neighbors=i>=0?topK(S.items[i].vec,Number($("kn").value),i):[];
  if(i<0)S.isolate=false;
  renderSelection();S.dirty=true;
}
function listInto(ul,rows,onPick){
  ul.innerHTML="";const cos=$("metric").value==="cosine";
  const maxD=rows.length?Math.max(...rows.map(r=>r[1]))*1.05||1:1;
  for(const [j,val] of rows){
    const li=document.createElement("li");const b=document.createElement("button");b.type="button";
    const lab=document.createElement("span");lab.className="lab";lab.textContent=S.items[j].label;lab.title=S.items[j].label;
    const v=document.createElement("span");v.className="val";v.textContent=val.toFixed(3);
    const bar=document.createElement("span");bar.className="bar";const fill=document.createElement("span");
    fill.style.width=Math.max(2,Math.min(100,(cos?Math.max(0,val):1-val/maxD)*100))+"%";bar.append(fill);
    b.append(lab,v,bar);b.onclick=()=>onPick(j);li.append(b);ul.append(li);
  }
}
function renderSelection(){
  const card=$("selCard");const i=S.selected;
  $("isolateBtn").disabled=i<0;$("isolateBtn").setAttribute("aria-pressed",String(S.isolate));
  $("nbControls").hidden=i<0;
  if(i<0){card.innerHTML='<p class="hint" style="margin:0">Click a point to see its nearest neighbors in the original space.</p>';$("nbList").innerHTML="";return;}
  const it=S.items[i];card.innerHTML="";
  const d=document.createElement("div");d.className="card";
  const t=document.createElement("div");t.textContent=it.label;d.append(t);
  const g=document.createElement("div");g.className="g";const dot_=document.createElement("i");dot_.style.background=pointColor(i);
  const gs=document.createElement("span");gs.textContent=(it.group?it.group+", ":"")+"point "+fmt(i+1);g.append(dot_,gs);d.append(g);
  card.append(d);
  listInto($("nbList"),S.neighbors,select);
}
function updateQueryUI(){
  const ok=!!S.embedder;$("queryInput").disabled=!ok;$("queryBtn").disabled=!ok;
  $("queryHint").textContent=ok?"Your text is embedded with the same method and appears as a diamond on the map.":"New text can’t be embedded for "+(S.source==="demo"?"the demo vectors":"imported vectors")+". Use search to find points by label.";
}
async function runQuery(){
  const text=$("queryInput").value.trim();if(!text||!S.embedder)return;
  $("queryBtn").disabled=true;$("queryHint").textContent="Embedding…";
  try{
    const v=await S.embedder.embed(text);
    const nb=topK(v,10,-1);
    S.query={text,vec:v,nbrs:nb,pos:null};placeQuery();
    listInto($("queryList"),nb,select);
    const appr=S.method==="tsne"||S.method==="umap";
    $("queryHint").textContent="Closest points to “"+text.slice(0,60)+"”."+(appr?" On this layout the diamond sits among its nearest neighbors, as an approximation.":"");
  }catch(e){$("queryHint").textContent="Couldn’t embed that text: "+errText(e);}
  finally{$("queryBtn").disabled=!S.embedder;S.dirty=true;}
}
function placeQuery(){
  const q=S.query;if(!q)return;q.pos=null;const f=S.fit;if(!f)return;
  const nd=effNd();const v=q.vec;
  if(S.method==="pca"&&S.pcaInfo){
    const P=S.pcaInfo;const c=P.comps.map(cv=>{let s=0;for(let k=0;k<cv.length;k++)s+=(v[k]-P.mean[k])*cv[k];return s;});
    q.pos=[0,0,0];for(let d=0;d<nd;d++)q.pos[d]=(c[d]-f.mean[d])/f.scale;
  }else if(S.method==="custom"&&S.customInfo){
    const {m,dirX,dirY,P}=S.customInfo;const nv=vnorm(v)||1;let x=0,y=0;
    for(let k=0;k<m.length;k++){const a=v[k]/nv;x+=(a-m[k])*dirX[k];if(dirY)y+=(a-m[k])*dirY[k];}
    if(P){y=0;for(let k=0;k<m.length;k++)y+=(v[k]/nv-P.mean[k])*P.comps[0][k];}
    q.pos=[(x-f.mean[0])/f.scale,(y-f.mean[1])/f.scale,0];
  }
}
function queryPos(){
  const q=S.query;if(!q)return null;if(q.pos)return q.pos;
  const nb=q.nbrs.slice(0,5);let w=0;const p=[0,0,0];
  for(const [j,val] of nb){const wt=Math.max(0.001,$("metric").value==="cosine"?val:1/(1e-6+val));for(let d=0;d<3;d++)p[d]+=S.xyz[j*3+d]*wt;w+=wt;}
  return w?p.map(x=>x/w):null;
}

// ---------- search ----------
function updateSearch(){
  const q=$("search").value.trim().toLowerCase();const ul=$("searchList");ul.innerHTML="";
  if(!q){S.matches=null;$("searchInfo").textContent="";S.dirty=true;return;}
  const m=new Set();S.items.forEach((it,i)=>{if((String(it.label)+" "+(it.group||"")).toLowerCase().includes(q))m.add(i);});
  S.matches=m;
  $("searchInfo").textContent=m.size?fmt(m.size)+(m.size===1?" match":" matches")+(m.size>50?", showing the first 50":""):"No points match.";
  [...m].slice(0,50).forEach(i=>{
    const li=document.createElement("li");const b=document.createElement("button");b.type="button";b.style.gridTemplateColumns="1fr";
    const lab=document.createElement("span");lab.className="lab";lab.textContent=S.items[i].label;b.append(lab);b.onclick=()=>select(i);li.append(b);ul.append(li);
  });
  S.dirty=true;
}

// ---------- drawing ----------
const canvas=$("plot");const ctx=canvas.getContext("2d");let W=0,H=0,DPR=1;
function resize(){
  const r=canvas.getBoundingClientRect();DPR=Math.min(window.devicePixelRatio||1,2);W=r.width;H=r.height;
  canvas.width=Math.max(1,Math.round(W*DPR));canvas.height=Math.max(1,Math.round(H*DPR));S.dirty=true;
}
new ResizeObserver(resize).observe(canvas);
function projector(){
  const v=S.view,nd=effNd();const base=Math.min(W,H)*0.40*v.zoom;const cx=W/2+v.panX,cy=H/2+v.panY;
  const cyw=Math.cos(v.yaw),syw=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch);
  return (x,y,z,out)=>{
    if(nd===3){const x1=x*cyw-z*syw;const z1=x*syw+z*cyw;const y1=y*cp-z1*sp;const z2=y*sp+z1*cp;const p=CAM/(CAM-z2);
      out[0]=cx+x1*base*p;out[1]=cy-y1*base*p;out[2]=z2;out[3]=p;}
    else{out[0]=cx+x*base;out[1]=cy-y*base;out[2]=0;out[3]=1;}
    return out;
  };
}
function halo(text,x,y,{bold=false,color=colors.ink,align="left"}={}){
  ctx.font=(bold?"600 ":"500 ")+"12.5px "+FONT;
  ctx.textAlign=align;ctx.textBaseline="middle";ctx.lineJoin="round";
  ctx.lineWidth=3.5;ctx.strokeStyle=colors.stage;ctx.globalAlpha=0.92;ctx.strokeText(text,x,y);
  ctx.globalAlpha=1;ctx.fillStyle=color;ctx.fillText(text,x,y);
}
const short=(s,n=42)=>{s=String(s);return s.length>n?s.slice(0,n-1)+"…":s;};
function draw(){
  ctx.setTransform(DPR,0,0,DPR,0,0);ctx.globalAlpha=1;ctx.fillStyle=colors.stage;ctx.fillRect(0,0,W,H);
  const n=S.items.length;if(!n||S.xyz.length<n*3)return;
  const P=projector();const nd=effNd();const tmp=[0,0,0,0];
  if(S.scr.length!==n*4){S.scr=new Float32Array(n*4);S.vis=new Uint8Array(n);S.order=new Int32Array(n);}
  const scr=S.scr;
  for(let i=0;i<n;i++){P(S.xyz[i*3],S.xyz[i*3+1],S.xyz[i*3+2],tmp);scr[i*4]=tmp[0];scr[i*4+1]=tmp[1];scr[i*4+2]=tmp[2];scr[i*4+3]=tmp[3];}
  // axes
  if(S.axisLabels&&(S.method==="pca"||S.method==="custom")){
    ctx.lineWidth=1;ctx.strokeStyle=colors.muted;
    for(let a=0;a<nd;a++){
      const e0=[0,0,0],e1=[0,0,0];e0[a]=-1.08;e1[a]=1.08;
      const p0=P(...e0,[0,0,0,0]),p1=P(...e1,[0,0,0,0]);
      ctx.globalAlpha=0.32;ctx.beginPath();ctx.moveTo(p0[0],p0[1]);ctx.lineTo(p1[0],p1[1]);ctx.stroke();ctx.globalAlpha=1;
      const [lo,hi]=S.axisLabels[a]||["",""];
      if(hi)halo(short(hi,30),p1[0]+(a===0&&nd===2?-4:4),p1[1]+(a===0&&nd===2?-10:0),{color:colors.muted,align:a===0&&nd===2?"right":"left"});
      if(lo)halo(short(lo,30),p0[0]+(a===0&&nd===2?4:4),p0[1]+(a===0&&nd===2?-10:0),{color:colors.muted,align:"left"});
    }
  }
  const sel=S.selected;const nbSet=new Set(S.neighbors.map(x=>x[0]));
  const order=S.order;for(let i=0;i<n;i++)order[i]=i;
  if(nd===3)order.sort((a,b)=>scr[a*4+2]-scr[b*4+2]);
  const iso=S.isolate&&sel>=0;
  const faded=(i)=>{
    if(S.matches&&!S.matches.has(i))return true;
    if(S.off.size){const k=legendKey(i);if(k&&S.off.has(k))return true;}
    return false;
  };
  // neighbor lines
  if(sel>=0){
    ctx.strokeStyle=colors.accent;ctx.lineWidth=1.2;ctx.globalAlpha=0.55;ctx.beginPath();
    for(const j of nbSet){ctx.moveTo(scr[sel*4],scr[sel*4+1]);ctx.lineTo(scr[j*4],scr[j*4+1]);}ctx.stroke();
  }
  const qp=queryPos();let qs=null;
  if(qp){qs=P(qp[0],qp[1],qp[2],[0,0,0,0]);ctx.setLineDash([4,4]);ctx.strokeStyle=colors.ink;ctx.globalAlpha=0.45;ctx.lineWidth=1;ctx.beginPath();
    for(const [j] of S.query.nbrs.slice(0,5)){ctx.moveTo(qs[0],qs[1]);ctx.lineTo(scr[j*4],scr[j*4+1]);}ctx.stroke();ctx.setLineDash([]);}
  // points
  const base=S.psize*0.62;
  for(let o=0;o<n;o++){
    const i=order[o];let vis=1;
    if(iso&&i!==sel&&!nbSet.has(i))vis=0;
    S.vis[i]=vis;if(!vis)continue;
    const fd=faded(i);const depthA=nd===3?0.42+0.58*Math.min(1,Math.max(0,(scr[i*4+2]+1)/2)):0.9;
    ctx.globalAlpha=fd?0.1:depthA;ctx.fillStyle=pointColor(i);
    const r=Math.max(1.2,base*scr[i*4+3]);
    ctx.beginPath();ctx.arc(scr[i*4],scr[i*4+1],r,0,6.2832);ctx.fill();
  }
  ctx.globalAlpha=1;
  // rings
  const ring=(i,w,c,extra)=>{const r=Math.max(1.2,base*scr[i*4+3])+extra;ctx.lineWidth=w;ctx.strokeStyle=c;ctx.beginPath();ctx.arc(scr[i*4],scr[i*4+1],r,0,6.2832);ctx.stroke();};
  for(const j of nbSet)ring(j,1.5,colors.accent,2);
  if(sel>=0)ring(sel,2.2,colors.ink,3.5);
  if(S.hover>=0&&S.hover!==sel&&S.vis[S.hover])ring(S.hover,1.5,colors.ink,2.5);
  if(qs){const s=7;ctx.fillStyle=colors.accent;ctx.strokeStyle=colors.ink;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(qs[0],qs[1]-s);ctx.lineTo(qs[0]+s,qs[1]);ctx.lineTo(qs[0],qs[1]+s);ctx.lineTo(qs[0]-s,qs[1]);ctx.closePath();ctx.fill();ctx.stroke();}
  // labels with collision check
  const boxes=[];
  const tryLabel=(i,force,bold)=>{
    if(!S.vis[i])return;const txt=short(S.items[i].label);
    ctx.font=(bold?"600 ":"500 ")+"12.5px "+FONT;
    const w=ctx.measureText(txt).width;const r=Math.max(1.2,base*scr[i*4+3]);
    const x=scr[i*4]+r+5,y=scr[i*4+1];const box=[x-2,y-8,x+w+2,y+8];
    if(!force)for(const b of boxes)if(box[0]<b[2]&&box[2]>b[0]&&box[1]<b[3]&&box[3]>b[1])return;
    boxes.push(box);halo(txt,x,y,{bold});
  };
  if(qs){halo("“"+short(S.query.text,36)+"”",qs[0]+11,qs[1],{bold:true,color:colors.accent});boxes.push([qs[0]+9,qs[1]-8,qs[0]+260,qs[1]+8]);}
  if(sel>=0)tryLabel(sel,true,true);
  if(S.hover>=0&&S.hover!==sel)tryLabel(S.hover,true,false);
  for(const j of nbSet)tryLabel(j,false,false);
  if(S.matches&&S.matches.size<=40)for(const i of S.matches)tryLabel(i,false,false);
  if(S.allLabels){let tries=0;for(let o=n-1;o>=0&&tries<1500;o--){const i=order[o];if(faded(i))continue;tries++;tryLabel(i,false,false);}}
}

// ---------- loop ----------
function frame(){
  requestAnimationFrame(frame);
  const L=S.layout;
  if(L&&!S.paused){
    const t0=performance.now();let going=true;
    do{going=stepLayout();}while(going&&performance.now()-t0<14);
    applyCoords(layoutCoords(),L.nd);
    if(!going){S.layout=null;setProgress("");$("pauseBtn").disabled=true;}
    else{setProgress((L.type==="tsne"?"t-SNE":"UMAP")+" step "+fmt(L.step)+" of "+fmt(L.total));$("pauseBtn").disabled=false;}
  }
  if(S.autoRotate&&effNd()===3&&!S.drag&&S.items.length){S.view.yaw+=0.0022;S.dirty=true;}
  if(S.dirty){S.dirty=false;draw();}
}

// ---------- interaction ----------
const pointers=new Map();let pinch=null;
function pick(mx,my){
  let best=-1,bd=Infinity;const n=S.items.length;const base=S.psize*0.62;
  for(let i=0;i<n;i++){
    if(!S.vis[i])continue;const dx=S.scr[i*4]-mx,dy=S.scr[i*4+1]-my;const d=dx*dx+dy*dy;
    const r=Math.max(9,base*S.scr[i*4+3]+4);
    if(d<r*r&&d-S.scr[i*4+2]*4<bd){bd=d-S.scr[i*4+2]*4;best=i;}
  }
  return best;
}
function showTip(i,mx,my){
  const tip=$("tip");if(i<0){tip.hidden=true;return;}
  const it=S.items[i];tip.innerHTML="";const a=document.createElement("div");a.textContent=short(it.label,140);tip.append(a);
  if(it.group){const s=document.createElement("small");s.textContent=it.group;tip.append(s);}
  tip.hidden=false;const tw=tip.offsetWidth,th=tip.offsetHeight;
  tip.style.left=Math.min(W-tw-8,mx+14)+"px";tip.style.top=Math.max(8,Math.min(H-th-8,my+14))+"px";
}
canvas.addEventListener("pointerdown",e=>{
  canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.offsetX,y:e.offsetY});
  if(pointers.size===2){const [a,b]=[...pointers.values()];pinch={d:Math.hypot(a.x-b.x,a.y-b.y)||1,zoom:S.view.zoom};}
  S.drag={moved:false,pan:e.shiftKey||e.button===2||effNd()===2,lx:e.offsetX,ly:e.offsetY,sx:e.offsetX,sy:e.offsetY};
  canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove",e=>{
  if(pointers.has(e.pointerId))pointers.set(e.pointerId,{x:e.offsetX,y:e.offsetY});
  if(pointers.size===2&&pinch){const [a,b]=[...pointers.values()];S.view.zoom=Math.max(0.2,Math.min(30,pinch.zoom*Math.hypot(a.x-b.x,a.y-b.y)/pinch.d));if(S.drag)S.drag.moved=true;S.dirty=true;return;}
  if(S.drag&&pointers.size){
    const dx=e.offsetX-S.drag.lx,dy=e.offsetY-S.drag.ly;S.drag.lx=e.offsetX;S.drag.ly=e.offsetY;
    if(Math.hypot(e.offsetX-S.drag.sx,e.offsetY-S.drag.sy)>4)S.drag.moved=true;
    if(S.drag.pan){S.view.panX+=dx;S.view.panY+=dy;}
    else{S.view.yaw+=dx*0.009;S.view.pitch=Math.max(-1.55,Math.min(1.55,S.view.pitch+dy*0.009));}
    $("tip").hidden=true;S.dirty=true;return;
  }
  const h=pick(e.offsetX,e.offsetY);if(h!==S.hover){S.hover=h;S.dirty=true;}
  showTip(h,e.offsetX,e.offsetY);canvas.style.cursor=h>=0?"pointer":"";
});
function endPointer(e){
  const wasClick=S.drag&&!S.drag.moved&&pointers.size===1&&e.type==="pointerup"&&e.button===0;
  pointers.delete(e.pointerId);if(pointers.size<2)pinch=null;
  if(wasClick){const i=pick(e.offsetX,e.offsetY);if(i>=0)select(i);else if(S.selected>=0)select(-1);}
  if(!pointers.size){S.drag=null;canvas.classList.remove("dragging");}
}
canvas.addEventListener("pointerup",endPointer);canvas.addEventListener("pointercancel",endPointer);
canvas.addEventListener("pointerleave",()=>{if(!S.drag){S.hover=-1;$("tip").hidden=true;S.dirty=true;}});
canvas.addEventListener("contextmenu",e=>e.preventDefault());
canvas.addEventListener("wheel",e=>{
  e.preventDefault();const f=Math.exp(-e.deltaY*(e.deltaMode===1?0.05:0.0015));
  const z0=S.view.zoom;const z1=Math.max(0.2,Math.min(30,z0*f));const r=z1/z0;
  const mx=e.offsetX-W/2,my=e.offsetY-H/2;S.view.panX=mx-(mx-S.view.panX)*r;S.view.panY=my-(my-S.view.panY)*r;S.view.zoom=z1;S.dirty=true;
},{passive:false});
canvas.addEventListener("dblclick",()=>{S.view={...DEFAULT_VIEW};S.dirty=true;});
canvas.addEventListener("keydown",e=>{
  const v=S.view;const nd=effNd();let used=true;
  if(e.key==="Escape")select(-1);
  else if(e.key==="+"||e.key==="=")v.zoom=Math.min(30,v.zoom*1.15);
  else if(e.key==="-")v.zoom=Math.max(0.2,v.zoom/1.15);
  else if(e.key.startsWith("Arrow")){
    const dx=e.key==="ArrowLeft"?-1:e.key==="ArrowRight"?1:0,dy=e.key==="ArrowUp"?-1:e.key==="ArrowDown"?1:0;
    if(nd===3&&!e.shiftKey){v.yaw+=dx*0.08;v.pitch=Math.max(-1.55,Math.min(1.55,v.pitch+dy*0.08));}else{v.panX-=dx*24;v.panY-=dy*24;}
  }else used=false;
  if(used){e.preventDefault();S.dirty=true;}
});

function setHud(msg){$("hud").textContent=msg;}
function updateHud(){
  const nd=effNd();const name={pca:"PCA",tsne:"t-SNE",umap:"UMAP",custom:"Custom axes"}[S.method];
  const how=nd===3?"Drag to rotate, shift-drag to pan, scroll to zoom.":"Drag to pan, scroll to zoom.";
  $("hud").innerHTML="";const s=document.createElement("strong");s.textContent=name+", "+nd+"D. ";$("hud").append(s,document.createTextNode(how+" Click a point for its neighbors."));
}

// ---------- controls ----------
document.querySelectorAll(".tabs [role=tab]").forEach(tab=>tab.addEventListener("click",()=>openTab(tab.id.replace("tab-",""))));
function openTab(name){
  document.querySelectorAll(".tabs [role=tab]").forEach(t=>t.setAttribute("aria-selected",String(t.id==="tab-"+name)));
  for(const p of ["write","import","samples"])$("panel-"+p).hidden=p!==name;
}
document.querySelectorAll("#methodSeg button").forEach(b=>b.addEventListener("click",()=>{S.method=b.dataset.m;runProjection();placeQuery();S.dirty=true;}));
document.querySelectorAll("#dimSeg button").forEach(b=>b.addEventListener("click",()=>{if(b.disabled)return;S.ndPref=Number(b.dataset.d);S.view={...DEFAULT_VIEW};runProjection();placeQuery();}));
for(const [id,out,fx] of [["perp","perpOut",v=>v],["lr","lrOut",v=>v],["nn","nnOut",v=>v],["md","mdOut",v=>Number(v).toFixed(2)],["psize","psizeOut",v=>v],["kSel","kOut",v=>v],["kn","knOut",v=>v]]){
  $(id).addEventListener("input",()=>{$(out).textContent=fx($(id).value);});
}
for(const id of ["perp","lr","nn","md"])$(id).addEventListener("change",()=>{runProjection();});
$("psize").addEventListener("input",()=>{S.psize=Number($("psize").value);S.dirty=true;});
$("kSel").addEventListener("change",()=>{S.k=Number($("kSel").value);S.off.clear();computeClusters();renderLegend();renderSelection();S.dirty=true;});
$("kn").addEventListener("change",()=>select(S.selected));
$("metric").addEventListener("change",()=>{select(S.selected);if(S.query){S.query.nbrs=topK(S.query.vec,10,-1);listInto($("queryList"),S.query.nbrs,select);}});
$("colorBy").addEventListener("change",()=>{S.colorBy=$("colorBy").value;S.autoCluster=false;$("kWrap").hidden=S.colorBy!=="cluster";S.off.clear();if(S.colorBy==="cluster"&&!S.clusters)computeClusters();renderLegend();renderSelection();S.dirty=true;});
$("allLabels").addEventListener("change",()=>{S.allLabels=$("allLabels").checked;S.dirty=true;});
$("autoRotate").checked=S.autoRotate;
$("autoRotate").addEventListener("change",()=>{S.autoRotate=$("autoRotate").checked;});
$("resetBtn").addEventListener("click",()=>{S.view={...DEFAULT_VIEW};S.dirty=true;});
$("isolateBtn").addEventListener("click",()=>{S.isolate=!S.isolate;$("isolateBtn").setAttribute("aria-pressed",String(S.isolate));S.dirty=true;});
$("rerunBtn").addEventListener("click",runProjection);
$("pauseBtn").addEventListener("click",()=>{if(!S.layout)return;S.paused=!S.paused;$("pauseBtn").textContent=S.paused?"Resume":"Pause";});
$("applyAxes").addEventListener("click",()=>{if(S.method==="custom")applyCustom();});
for(const id of ["xl","xr","yd","yu"])$(id).addEventListener("keydown",e=>{if(e.key==="Enter")applyCustom();});
$("search").addEventListener("input",updateSearch);
$("queryBtn").addEventListener("click",runQuery);
$("queryInput").addEventListener("keydown",e=>{if(e.key==="Enter")runQuery();});
$("themeBtn").addEventListener("click",()=>{
  const next=isDark()?"light":"dark";document.documentElement.setAttribute("data-theme",next);
  try{localStorage.setItem("va:theme",next);}catch(_){}
  readColors();
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",readColors);

// method selector
function methodHint(){
  const m=$("embedMethod").value;$("dimWrap").hidden=m!=="lexical";$("modelWrap").hidden=m!=="neural";
  const md=MODELS[$("modelSel").value];
  $("methodHint").textContent=m==="neural"
    ?"Runs "+md.name+" ("+md.languages+") on your device with transformers.js. The first run downloads "+md.size+"; after that the browser keeps it cached. Nothing you type leaves your computer."
    :"Instant and fully offline. Lines that share words or word pieces land close together, but synonyms don’t. Use the neural model for meaning.";
}
$("embedMethod").addEventListener("change",methodHint);
$("modelSel").addEventListener("change",methodHint);

// write tab
function parseLines(){
  return $("textInput").value.split(/\r?\n/).map(l=>l.trim()).filter(Boolean).map(l=>{
    const m=l.match(/^\[([^\]]{1,40})\]\s*(.+)$/);return m?{group:m[1].trim(),text:m[2].trim()}:{group:null,text:l};
  });
}
let embedCtl=null;
function downloadProgress(st){
  const files=new Map();
  return (ev)=>{
    if(ev.status==="progress"&&ev.file){files.set(ev.file,{loaded:ev.loaded||0,total:ev.total||0});
      let l=0,t=0;for(const f of files.values()){l+=f.loaded;t+=f.total;}
      st.textContent="Downloading the model: "+(l/1e6).toFixed(1)+" of "+(t/1e6).toFixed(1)+" MB…";}
    else if(ev.status==="ready")st.textContent="Model ready. Embedding…";
  };
}
$("embedBtn").addEventListener("click",async()=>{
  const st=$("embedStatus");st.className="status";
  const rows=parseLines();
  if(rows.length<2){st.className="status err";st.textContent="Add at least two lines of text first.";return;}
  const method=$("embedMethod").value;
  const toItems=(V)=>rows.map((r,i)=>({label:r.text,text:r.text,group:r.group,vec:V[i]}));
  if(method==="lexical"){
    if(rows.length>5000){st.className="status err";st.textContent="That’s "+fmt(rows.length)+" lines. The limit here is 5,000.";return;}
    const dim=Number($("dimSel").value);
    const E=new LexicalEmbedder(rows.map(r=>r.text),{dim});
    setData(toItems(E.embedAll()),{source:"lexical",embedder:E,dim});
    st.textContent="Embedded "+fmt(rows.length)+" lines.";return;
  }
  if(rows.length>3000){st.className="status err";st.textContent="The neural model handles up to 3,000 lines here. You have "+fmt(rows.length)+".";return;}
  embedCtl=new AbortController();$("embedBtn").disabled=true;$("stopEmbedBtn").hidden=false;
  try{
    const key=$("modelSel").value;
    st.textContent="Loading the model…";
    const extractor=await loadModel(key,downloadProgress(st));
    const E=new NeuralEmbedder(key,extractor);
    const V=await E.embedMany(rows.map(r=>r.text),{signal:embedCtl.signal,onProgress:(d,t)=>{st.textContent="Embedding lines: "+fmt(d)+" of "+fmt(t)+"…";}});
    rows.forEach((r,i)=>E.cache.set(r.text,V[i]));
    setData(toItems(V),{source:"neural",embedder:E,dim:E.dim});
    st.textContent="Embedded "+fmt(rows.length)+" lines with "+E.model.name+".";
  }catch(e){
    st.className=e&&e.name==="AbortError"?"status":"status err";
    st.textContent=e&&e.name==="AbortError"?"Stopped.":"The model couldn’t run: "+errText(e)+". Check your connection, or switch to words and subwords.";
  }
  finally{$("embedBtn").disabled=false;$("stopEmbedBtn").hidden=true;embedCtl=null;}
});
$("stopEmbedBtn").addEventListener("click",()=>{if(embedCtl)embedCtl.abort();});
let saveT=null;
$("textInput").addEventListener("input",()=>{clearTimeout(saveT);saveT=setTimeout(()=>{try{localStorage.setItem("va:text",$("textInput").value);}catch(_){}},400);});

// import tab
function readFileInto(input,target){
  input.addEventListener("change",()=>{const f=input.files&&input.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{$(target).value=r.result;};r.readAsText(f);});
}
readFileInto($("vecFile"),"vecInput");readFileInto($("metaFile"),"metaInput");
function itemsFromImport(rows,meta,labelCol,groupCol){
  return rows.map((r,i)=>{
    const mrow=meta?meta.rows[i]||[]:[];
    const label=r.label??(meta?mrow[labelCol]:undefined)??("Point "+(i+1));
    const group=r.group??(meta&&groupCol>=0?mrow[groupCol]:null);
    return {label:String(label),text:String(label),group:group==null||group===""?null:String(group),vec:Float32Array.from(r.vector)};
  });
}
let lastImport=null;
$("importBtn").addEventListener("click",()=>{
  const st=$("importStatus");st.className="status err";
  try{
    const rows=parseVectors($("vecInput").value);
    if(rows.length<2)throw new Error("Add at least two vectors.");
    const d=rows[0].vector.length;if(d<2)throw new Error("Each vector needs at least two values.");
    const bad=rows.findIndex(r=>r.vector.length!==d);
    if(bad>=0)throw new Error("Row "+(bad+1)+" has "+rows[bad].vector.length+" values, but row 1 has "+d+".");
    const meta=parseMeta($("metaInput").value);
    if(meta&&meta.rows.length!==rows.length)throw new Error("Metadata has "+fmt(meta.rows.length)+" rows but there are "+fmt(rows.length)+" vectors.");
    let labelCol=0,groupCol=-1;
    if(meta&&meta.headers.length>1){
      const h=meta.headers.map(x=>x.toLowerCase());
      labelCol=Math.max(0,h.findIndex(x=>["text","label","name","word","title"].includes(x)));
      groupCol=h.findIndex(x=>["group","category","class","label","topic","cluster","type"].includes(x)&&h.indexOf(x)!==labelCol);
      if(groupCol<0)groupCol=labelCol===0?1:0;
      for(const [sel,val] of [["labelCol",labelCol],["groupCol",groupCol]]){
        const s=$(sel);s.innerHTML="";meta.headers.forEach((name,i)=>{const o=document.createElement("option");o.value=i;o.textContent=name;s.append(o);});
        if(sel==="groupCol"){const o=document.createElement("option");o.value=-1;o.textContent="None";s.prepend(o);}
        s.value=val;
      }
      $("metaCols").hidden=false;
    }else $("metaCols").hidden=true;
    lastImport={rows,meta};
    setData(itemsFromImport(rows,meta,labelCol,groupCol),{source:"imported",embedder:null,dim:d,meta});
    st.className="status";st.textContent="Loaded "+fmt(rows.length)+" vectors with "+fmt(d)+" dimensions.";
  }catch(err){st.textContent=err.message||String(err);}
});
for(const id of ["labelCol","groupCol"])$(id).addEventListener("change",()=>{
  if(!lastImport||S.source!=="imported")return;
  const items=itemsFromImport(lastImport.rows,lastImport.meta,Number($("labelCol").value),Number($("groupCol").value));
  items.forEach((it,i)=>{S.items[i].label=it.label;S.items[i].text=it.text;S.items[i].group=it.group;});
  S.off.clear();computeGroups();renderSelection();updateSearch();
});

// samples
document.querySelectorAll(".sample").forEach(b=>b.addEventListener("click",()=>{
  const k=b.dataset.sample;
  if(k==="concepts"){loadDemo();return;}
  $("textInput").value=k==="topics"?SAMPLE_TOPICS:SAMPLE_REVIEWS;
  try{localStorage.setItem("va:text",$("textInput").value);}catch(_){}
  openTab("write");
  $("embedStatus").className="status";$("embedStatus").textContent="Sample added. Pick a method, then press Embed and plot.";
}));
function loadDemo(){
  const rows=buildDemo();
  setData(rows.map(r=>({label:r.label,text:r.label,group:r.group,vec:r.vector})),{source:"demo",embedder:null,dim:48});
}

// export
function save(filename,data,type){
  if(!S.items.length)return;
  const url=URL.createObjectURL(new Blob([data],{type}));
  const link=document.createElement("a");link.href=url;link.download=filename;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
  $("exportStatus").textContent="Saved "+filename+".";
}
$("expVec").addEventListener("click",()=>save("embeddings.json",JSON.stringify(S.items.map(it=>({text:it.label,group:it.group,vector:Array.from(it.vec,x=>Math.round(x*1e6)/1e6)}))),"application/json"));
$("expXY").addEventListener("click",()=>{
  const nd=effNd();const q=(s)=>'"'+String(s??"").replace(/"/g,'""')+'"';
  const head=["label","group","x","y"].concat(nd===3?["z"]:[]).join(",");
  const lines=S.items.map((it,i)=>[q(it.label),q(it.group)].concat([0,1,2].slice(0,nd).map(d=>S.xyz[i*3+d].toFixed(5))).join(","));
  save("coordinates.csv",[head,...lines].join("\n"),"text/csv");
});

// ---------- start ----------
(function init(){
  try{const t=localStorage.getItem("va:theme");if(t)document.documentElement.setAttribute("data-theme",t);}catch(_){}
  readColors();
  let saved=null;try{saved=localStorage.getItem("va:text");}catch(_){}
  $("textInput").value=saved||SAMPLE_TOPICS;
  methodHint();resize();loadDemo();
  if(document.fonts&&document.fonts.ready)document.fonts.ready.then(()=>{S.dirty=true;});
  requestAnimationFrame(frame);
})();
