/* eslint-disable @next/next/no-img-element */
interface Props {
  preview: string;
  onReset: () => void;
}

export default function PreviewImage({ preview, onReset }: Props) {
  return (
    <div className="space-y-4 flex flex-col items-center">
      <img
        src={preview}
        alt="Preview"
        className="rounded-xl object-cover w-full max-w-62.5 aspect-square"
      />

      <button
        onClick={onReset}
        className="text-sm text-gray-500 hover:text-black transition"
      >
        Change Image
      </button>
    </div>
  );
}
