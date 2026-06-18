import { PredictionResponse } from "@/types/prediction";

export async function analyzeImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();

  formData.append("file", file);
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Backend request failed");
  }

  return response.json();
}
