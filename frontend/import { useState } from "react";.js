import { useState } from "react";

export default function Encoding(){
  const [base, setBase] = useState(64);
  const [hex, setHex] = useState("48656c6c6f2051754c6162"); // "Hello QuLab"
  const [encoded, setEncoded] = useState("");
  const [decodedHex, setDecodedHex] = useState("");

  const encode = async () => {
    const res = await fetch("http://localhost:8000/encoding/encode", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({base, data_hex: hex})
    });
    const j = await res.json();
    setEncoded(j.encoded);
  };

  const decode = async () => {
    const res = await fetch("http://localhost:8000/encoding/decode", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({base, encoded})
    });
    const j = await res.json();
    setDecodedHex(j.data_hex);
  };

  return (
    <div className="card">
      <h1>High-Radix Encoding (Lite)</h1>
      <div style={{display:"grid", gap:8, gridTemplateColumns:"1fr 1fr"}}>
        <div>
          <label>Base (2..91): </label>
          <input className="input" type="number" value={base} onChange={e=>setBase(parseInt(e.target.value||"64"))}/>
        </div>
        <div>
          <label>HEX input:</label>
          <textarea className="input" rows={3} value={hex} onChange={e=>setHex(e.target.value)} />
        </div>
      </div>
      <div style={{display:"flex", gap:8, marginTop:8}}>
        <button className="btn" onClick={encode}>Encode</button>
        <button className="btn" onClick={decode} disabled={!encoded}>Decode</button>
      </div>
      <div style={{marginTop:12}}>
        <div><b>Encoded:</b> {encoded}</div>
        <div><b>Decoded HEX:</b> {decodedHex}</div>
      </div>
    </div>
  );
}