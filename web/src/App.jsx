import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: form,
      });
      setResult(await res.json());
    } catch {
      setResult({ prediction: "Error connecting to server", confidence: 0 });
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Onion Classifier</h1>
      <p className="subtitle">Upload an onion image to check its quality</p>

      <label className="upload-area" htmlFor="file-input">
        {preview ? (
          <img src={preview} alt="preview" className="preview" />
        ) : (
          <span className="upload-text">Click to upload an image</span>
        )}
        <input id="file-input" type="file" accept="image/*" onChange={handleFile} hidden />
      </label>

      <button className="classify-btn" onClick={handleSubmit} disabled={!file || loading}>
        {loading ? "Classifying..." : "Classify"}
      </button>

      {result && (
        <div className={`result ${result.prediction === "good onion" ? "good" : "bad"}`}>
          <span className="prediction">{result.prediction}</span>
          <span className="confidence">{result.confidence}% confidence</span>
        </div>
      )}
    </div>
  );
}

export default App;
