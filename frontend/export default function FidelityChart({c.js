export default function FidelityChart({counts}:{counts:Record<string,number>}){
  const entries = Object.entries(counts);
  const total = entries.reduce((a, [,v])=>a+v,0);
  return (
    <div className="card" style={{marginTop:12}}>
      <b>Counts</b>
      <div style={{display:"flex", gap:8, marginTop:6}}>
        {entries.map(([k,v])=>(
          <div key={k} title={`${k}: ${v}`}>
            <div style={{height: Math.max(8, (v/Math.max(1,total))*120), width: 24, background:"#1e40af", borderRadius:4}}/>
            <div style={{textAlign:"center"}}>{k}</div>
          </div>
        ))}
      </div>
    </div>
  );
}