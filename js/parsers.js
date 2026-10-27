// Parsers for pasted or uploaded vectors and metadata.

// Parse pasted vectors: TSV/CSV/space separated, optional leading label, or JSON
export function parseVectors(text){
  const t=text.trim();if(!t)throw new Error("Paste vectors or choose a file first.");
  if(t[0]==="["||t[0]==="{"){
    let data=JSON.parse(t);if(!Array.isArray(data))data=data.items||data.data||data.embeddings||[];
    const rows=data.map((r,i)=>{
      if(Array.isArray(r))return {vector:r.map(Number)};
      const vec=r.vector||r.embedding||r.values||r.vec;if(!Array.isArray(vec))throw new Error("Item "+(i+1)+" has no vector, embedding or values array.");
      return {vector:vec.map(Number),label:r.text??r.label??r.name??r.word,group:r.group??r.category??r.class??r.cluster};
    });
    return rows;
  }
  const lines=t.split(/\r?\n/).filter(l=>l.trim());
  const sep=lines[0].includes("\t")?"\t":(lines[0].includes(",")?",":/\s+/);
  return lines.map((l,i)=>{
    const parts=l.trim().split(sep).map(s=>s.trim()).filter(s=>s!=="");
    let label;
    if(parts.length&&isNaN(Number(parts[0]))){label=parts.shift();}
    const vec=parts.map(Number);
    if(vec.some(x=>!isFinite(x)))throw new Error("Line "+(i+1)+" has a value that is not a number.");
    return {vector:vec,label};
  });
}
export function parseMeta(text,n){
  const lines=text.replace(/\s+$/,"").split(/\r?\n/);if(!text.trim())return null;
  const multi=lines[0].includes("\t");
  if(!multi)return {headers:["label"],rows:lines.map(l=>[l.trim()])};
  const headers=lines[0].split("\t").map(s=>s.trim()||"column");
  return {headers,rows:lines.slice(1).map(l=>l.split("\t").map(s=>s.trim()))};
}

