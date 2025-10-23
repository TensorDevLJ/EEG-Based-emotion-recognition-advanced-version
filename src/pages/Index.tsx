import { useState } from "react";
import { Upload, Brain, FileText, BarChart3, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import FileUpload from "@/components/FileUpload";
import ResultsDashboard from "@/components/ResultsDashboard";

const Index = () => {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const handleAnalyze = async () => {
    if (!file) {
      toast({
        title: "No file selected",
        description: "Please upload an EEG file first",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResults(data);
      
      toast({
        title: "Analysis Complete",
        description: "Your EEG data has been analyzed successfully",
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-muted/30 to-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                <Brain className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  EEG Depression Analysis
                </h1>
                <p className="text-sm text-muted-foreground">Transformer-based Clinical Assessment</p>
              </div>
            </div>
            
            <Badge variant="secondary" className="hidden md:flex gap-2">
              <Activity className="w-4 h-4" />
              AI-Powered Analysis
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {!results ? (
          <div className="max-w-4xl mx-auto space-y-6">
            {/* Hero Section */}
            <Card className="border-2 bg-gradient-to-br from-card via-card to-muted/20">
              <CardHeader className="text-center space-y-4 pb-8">
                <div className="mx-auto w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center mb-4">
                  <Brain className="w-12 h-12 text-primary" />
                </div>
                <CardTitle className="text-3xl md:text-4xl font-bold">
                  Advanced EEG Depression Detection
                </CardTitle>
                <CardDescription className="text-base max-w-2xl mx-auto">
                  Upload EEG recordings for comprehensive analysis using state-of-the-art Transformer models. 
                  Get detailed insights into depression indicators with explainable AI.
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-8">
                <FileUpload 
                  file={file} 
                  setFile={setFile}
                  loading={loading}
                />

                <Button 
                  onClick={handleAnalyze}
                  disabled={!file || loading}
                  size="lg"
                  className="w-full h-14 text-lg font-semibold bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-opacity"
                >
                  {loading ? (
                    <>
                      <Activity className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing EEG Data...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="w-5 h-5 mr-2" />
                      Analyze Recording
                    </>
                  )}
                </Button>

                {loading && (
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm text-muted-foreground">
                      <span>Processing pipeline...</span>
                      <span>This may take a few moments</span>
                    </div>
                    <Progress value={33} className="h-2" />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Features Grid */}
            <div className="grid md:grid-cols-3 gap-4">
              <Card className="border-primary/20">
                <CardHeader>
                  <FileText className="w-8 h-8 text-primary mb-2" />
                  <CardTitle className="text-lg">Comprehensive Features</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  60+ spectral, temporal, and connectivity features extracted from your EEG data
                </CardContent>
              </Card>

              <Card className="border-secondary/20">
                <CardHeader>
                  <Brain className="w-8 h-8 text-secondary mb-2" />
                  <CardTitle className="text-lg">Transformer AI</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  Advanced deep learning model trained for accurate depression stage classification
                </CardContent>
              </Card>

              <Card className="border-accent/20">
                <CardHeader>
                  <BarChart3 className="w-8 h-8 text-accent mb-2" />
                  <CardTitle className="text-lg">Visual Insights</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  Interactive plots showing PSD, scalograms, connectivity matrices, and more
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          <ResultsDashboard results={results} onReset={() => { setResults(null); setFile(null); }} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t mt-16 py-6 bg-card/30">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>EEG Depression Analysis System • Research & Educational Purposes</p>
          <p className="mt-2">Always consult healthcare professionals for clinical decisions</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
