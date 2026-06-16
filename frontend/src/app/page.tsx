"use client";

import { useState } from "react";

import UploadBox from "@/components/UploadBox";
import PreviewImage from "@/components/PreviewImage";
import ResultCard from "@/components/ResultCard";
import { toast } from "sonner";
import { analyzeImage } from "@/services/api";
import Navbar from "@/components/Navbar";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);

  const [loading, setLoading] = useState(false);

  const handleFileSelect = (file: File) => {
    setFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleReset = () => {
    setFile(null);
    setPreview("");
    setPrediction("");
    setConfidence(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);

    try {
      const data = await analyzeImage(file);

      setPrediction(data.prediction);
      setConfidence(data.confidence);
    } catch (error) {
      console.error(error);
      toast.error("Backend connection failed");
    }

    setLoading(false);
  };

  return (
    <>
      <Navbar />
      <main className="min-h-screen bg-linear-to-br from-zinc-200 via-gray-200 to-zinc-200 flex items-center justify-center p-6">
        <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-lg space-y-6">
          <div className="text-center">
            <h1 className="text-3xl font-bold">Skin Lesion Classifier</h1>

            <p className="text-gray-500 mt-2">
              Upload a skin image for AI analysis
            </p>
          </div>

          {!preview ? (
            <UploadBox onFileSelect={handleFileSelect} />
          ) : (
            <PreviewImage preview={preview} onReset={handleReset} />
          )}

          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="w-full bg-black text-white py-3 rounded-xl hover:opacity-90 transition disabled:opacity-50"
          >
            {loading ? (
              <div className="flex items-center justify-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Analyzing...
              </div>
            ) : !file ? (
              "Upload an image first"
            ) : (
              "Analyze Image"
            )}
          </button>

          {prediction && confidence !== null && (
            <ResultCard prediction={prediction} confidence={confidence} />
          )}
        </div>
      </main>
    </>
  );
}
