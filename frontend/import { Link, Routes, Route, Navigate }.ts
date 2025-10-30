import { Link, Routes, Route, Navigate } from "react-router-dom";
import Teleport from "./routes/Teleport";
import Encoding from "./routes/Encoding";
import Governance from "./routes/Governance";

export default function App() {
  return (
    <>
      <nav className="nav">
        <Link to="/teleport">Teleport</Link>
        <Link to="/encoding">Encoding</Link>
        <Link to="/governance">Governance</Link>
        <a href="http://localhost:8000/health" target="_blank">API</a>
      </nav>
      <div className="container">
        <Routes>
          <Route path="/" element={<Navigate to="/teleport" />} />
          <Route path="/teleport" element={<Teleport/>} />
          <Route path="/encoding" element={<Encoding/>} />
          <Route path="/governance" element={<Governance/>} />
        </Routes>
      </div>
    </>
  );
}