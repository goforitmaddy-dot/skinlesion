import { ScanSearch } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="w-full bg-white/80 backdrop-blur-md shadow-sm px-6 py-4">
      <div className="max-w-6xl mx-auto grid grid-cols-3 items-center">
        <div className="flex items-center">
          <ScanSearch size={28} className="text-black" />
        </div>

        <div className="flex justify-center">
          <h1 className="text-xl font-semibold tracking-tight">Skinlesion</h1>
        </div>
        <div />
      </div>
    </nav>
  );
}
