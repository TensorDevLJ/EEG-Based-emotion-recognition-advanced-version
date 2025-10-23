import { Upload, FileCheck, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface FileUploadProps {
  file: File | null;
  setFile: (file: File | null) => void;
  loading: boolean;
}

const FileUpload = ({ file, setFile, loading }: FileUploadProps) => {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <div className="space-y-4">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all ${
          file 
            ? "border-success bg-success/5" 
            : "border-muted-foreground/30 hover:border-primary hover:bg-muted/50"
        } ${loading ? "opacity-50 pointer-events-none" : ""}`}
      >
        <input
          type="file"
          accept=".csv,.edf"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={loading}
        />
        
        <div className="space-y-4">
          {file ? (
            <>
              <div className="mx-auto w-16 h-16 rounded-full bg-success/20 flex items-center justify-center">
                <FileCheck className="w-8 h-8 text-success" />
              </div>
              <div>
                <p className="font-semibold text-lg">{file.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                }}
                className="gap-2"
              >
                <X className="w-4 h-4" />
                Remove File
              </Button>
            </>
          ) : (
            <>
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <Upload className="w-8 h-8 text-primary" />
              </div>
              <div>
                <p className="font-semibold text-lg">Upload EEG Recording</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Drag and drop or click to browse
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Supports .csv and .edf formats
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
