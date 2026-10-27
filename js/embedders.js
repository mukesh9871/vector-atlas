// Text embedders. Both expose embed(text) and return unit-length Float32Arrays.
import {fnv1a,normalizeInPlace} from "./math.js";

// ---------- Local lexical embedding: TF-IDF over words, word pairs and subword pieces, hashed into a fixed vector ----------
const STOP=new Set(("a an the and or but if of to in on at by for with from as is are was were be been being it its this that these those i im you he she we they me my your our their his her them us do does did have has had not no so than too very can could will would just into about over under then there here what which who whom when where why how all any each few more most other some such only own same also get got am").split(" "));
export function tokenize(text){return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu)||[]);}
export function featuresOf(text,opts){
  const feats=new Map();const add=(f,w)=>feats.set(f,(feats.get(f)||0)+w);
  const content=tokenize(text).filter(t=>!STOP.has(t)&&t.length>1);
  for(const t of content){
    add("w:"+t,1);
    if(opts.subwords&&t.length>3){const s="<"+t+">";for(let n=3;n<=5;n++)for(let i=0;i+n<=s.length;i++)add("c:"+s.slice(i,i+n),0.22);}
  }
  for(let i=0;i+1<content.length;i++)add("b:"+content[i]+"_"+content[i+1],0.5);
  return feats;
}
export class LexicalEmbedder{
  constructor(texts,{dim=512,subwords=true}={}){
    this.kind="lexical";this.dim=dim;this.opts={subwords};this.N=texts.length;this.df=new Map();
    this.docFeats=texts.map(t=>featuresOf(t,this.opts));
    for(const f of this.docFeats)for(const k of f.keys())this.df.set(k,(this.df.get(k)||0)+1);
  }
  idf(k){const d=this.df.get(k)||0;return Math.log((1+this.N)/(1+d))+1;}
  vec(feats){
    const v=new Float32Array(this.dim);
    for(const [k,w] of feats){
      const h=fnv1a(k);const idx=h%this.dim;const sign=(fnv1a("~"+k)&1)?1:-1;
      const tf=w>1?1+Math.log(w):w;
      v[idx]+=sign*tf*this.idf(k);
    }
    return normalizeInPlace(v);
  }
  embedAll(){return this.docFeats.map(f=>this.vec(f));}
  embed(text){return this.vec(featuresOf(text,this.opts));}
}


// ---------- Neural embedding with transformers.js (ONNX Runtime Web, runs on the visitor's device) ----------
export const MODELS={
  minilm:{id:"Xenova/all-MiniLM-L6-v2",name:"all-MiniLM-L6-v2",dim:384,size:"about 23 MB",languages:"English"},
  multilingual:{id:"Xenova/paraphrase-multilingual-MiniLM-L12-v2",name:"paraphrase-multilingual-MiniLM-L12-v2",dim:384,size:"about 120 MB",languages:"50+ languages"},
};
// Pinned on purpose: a fixed, well-tested release keeps the demo reproducible.
const TRANSFORMERS_URL="https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";
const pipelines=new Map();

export function loadModel(key,onProgress){
  if(!pipelines.has(key)){
    const p=(async()=>{
      const {pipeline,env}=await import(TRANSFORMERS_URL);
      env.allowLocalModels=false;
      return pipeline("feature-extraction",MODELS[key].id,{dtype:"q8",progress_callback:onProgress});
    })();
    p.catch(()=>pipelines.delete(key));
    pipelines.set(key,p);
  }
  return pipelines.get(key);
}

export class NeuralEmbedder{
  constructor(key,extractor){this.kind="neural";this.key=key;this.model=MODELS[key];this.dim=this.model.dim;this.extractor=extractor;this.cache=new Map();}
  async embedMany(texts,{batch=16,onProgress,signal}={}){
    const out=[];
    for(let i=0;i<texts.length;i+=batch){
      if(signal&&signal.aborted)throw new DOMException("Stopped.","AbortError");
      const tensor=await this.extractor(texts.slice(i,i+batch),{pooling:"mean",normalize:true});
      for(const row of tensor.tolist())out.push(Float32Array.from(row));
      if(onProgress)onProgress(Math.min(i+batch,texts.length),texts.length);
      await new Promise(r=>setTimeout(r,0)); // let the page repaint between batches
    }
    return out;
  }
  async embed(text){
    if(!this.cache.has(text)){const [v]=await this.embedMany([text]);this.cache.set(text,v);}
    return this.cache.get(text);
  }
}
