"use client";

import { useDropzone } from "react-dropzone";
import { UploadCloud } from "lucide-react";

interface Props {
  onFileSelect: (file: File) => void;
}

export default function UploadBox({ onFileSelect }: Props) {
  const onDrop = (acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];

    if (selectedFile) {
      onFileSelect(selectedFile);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [],
    },
    multiple: false,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition
        ${
          isDragActive ? "border-black bg-gray-100" : "border-gray-300 bg-white"
        }
      `}
    >
      <input {...getInputProps()} />

      <div className="flex flex-col items-center gap-4">
        <UploadCloud size={48} className="text-gray-400" />

        <div>
          <p className="text-gray-700 font-medium">
            Drag & drop your image here
          </p>

          <p className="text-sm text-gray-400 mt-1">or click to browse files</p>
        </div>
      </div>
    </div>
  );
}
