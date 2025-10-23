import { ArrowLeft, AlertCircle, CheckCircle, TrendingUp, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface ResultsDashboardProps {
  results: any;
  onReset: () => void;
}

const ResultsDashboard = ({ results, onReset }: ResultsDashboardProps) => {
  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "not depressed": return "secondary";
      case "mild": return "outline";
      case "moderate": return "outline";
      case "severe": return "destructive";
      default: return "secondary";
    }
  };

  const getSeverityIcon = (severity: string) => {
    if (severity.toLowerCase() === "not depressed") {
      return <CheckCircle className="w-5 h-5" />;
    }
    return <AlertCircle className="w-5 h-5" />;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={onReset} className="gap-2">
          <ArrowLeft className="w-4 h-4" />
          New Analysis
        </Button>
        <Badge variant="secondary" className="gap-2">
          <Activity className="w-4 h-4" />
          {results.subject_id}
        </Badge>
      </div>

      {/* Summary Cards */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card className="border-2">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Prediction</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {getSeverityIcon(results.predicted_class)}
              <Badge variant={getSeverityColor(results.predicted_class)} className="text-base px-3 py-1">
                {results.predicted_class}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Depression Index</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-2xl font-bold">{(results.depression_index * 100).toFixed(1)}%</div>
              <Progress value={results.depression_index * 100} className="h-2" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Dominant Wave</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">{results.dominant_wave}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-secondary">
              {(Math.max(...Object.values(results.probabilities) as number[]) * 100).toFixed(1)}%
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Explanation */}
      <Alert className="bg-muted/50 border-primary/20">
        <AlertCircle className="h-4 w-4 text-primary" />
        <AlertDescription className="ml-2">
          <div className="prose prose-sm max-w-none">
            <p className="whitespace-pre-line">{results.explanation_text}</p>
          </div>
        </AlertDescription>
      </Alert>

      {/* Probabilities */}
      <Card>
        <CardHeader>
          <CardTitle>Class Probabilities</CardTitle>
          <CardDescription>Model confidence for each depression stage</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Object.entries(results.probabilities).map(([className, prob]) => (
              <div key={className} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">{className}</span>
                  <span className="text-muted-foreground">{((prob as number) * 100).toFixed(1)}%</span>
                </div>
                <Progress value={(prob as number) * 100} className="h-2" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Top Features */}
      <Card>
        <CardHeader>
          <CardTitle>Top Contributing Features</CardTitle>
          <CardDescription>Features with highest impact on prediction (SHAP values)</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {results.top_features.map((feat: any, idx: number) => (
              <div key={idx} className="border rounded-lg p-4 space-y-2">
                <div className="flex items-start justify-between">
                  <div className="space-y-1 flex-1">
                    <div className="font-medium">{feat.feature}</div>
                    <div className="text-sm text-muted-foreground">
                      Value: {feat.value.toFixed(4)} • {feat.interpretation}
                    </div>
                  </div>
                  <Badge variant="secondary">
                    SHAP: {feat.shap_value.toFixed(3)}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Visualizations */}
      <Card>
        <CardHeader>
          <CardTitle>EEG Analysis Visualizations</CardTitle>
          <CardDescription>Comprehensive signal analysis and feature representations</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="time" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="time">Time Series</TabsTrigger>
              <TabsTrigger value="psd">PSD</TabsTrigger>
              <TabsTrigger value="bands">Band Power</TabsTrigger>
              <TabsTrigger value="scalogram">Scalogram</TabsTrigger>
              <TabsTrigger value="features">Features</TabsTrigger>
            </TabsList>

            {Object.entries(results.graphs).map(([key, base64]) => {
              const tabValue = key.replace('_plot', '').replace('_', '');
              return (
                <TabsContent key={key} value={tabValue} className="mt-6">
                  <div className="rounded-lg border bg-muted/20 p-4">
                    <img 
                      src={`data:image/png;base64,${base64}`} 
                      alt={key}
                      className="w-full h-auto rounded"
                    />
                  </div>
                </TabsContent>
              );
            })}

            <TabsContent value="features" className="mt-6">
              <div className="rounded-lg border bg-muted/20 p-4">
                {results.graphs.feature_importance_shap && (
                  <img 
                    src={`data:image/png;base64,${results.graphs.feature_importance_shap}`} 
                    alt="Feature Importance"
                    className="w-full h-auto rounded"
                  />
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Model Metrics */}
      {results.metrics && Object.keys(results.metrics).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Model Performance Metrics</CardTitle>
            <CardDescription>Test set evaluation results</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-4">
              {results.metrics.accuracy && (
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">Accuracy</div>
                  <div className="text-2xl font-bold">{(results.metrics.accuracy * 100).toFixed(1)}%</div>
                </div>
              )}
              {results.metrics.f1_macro && (
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">F1 Score (Macro)</div>
                  <div className="text-2xl font-bold">{(results.metrics.f1_macro * 100).toFixed(1)}%</div>
                </div>
              )}
              {results.metrics.roc_auc_macro && (
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">ROC AUC</div>
                  <div className="text-2xl font-bold">{(results.metrics.roc_auc_macro * 100).toFixed(1)}%</div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ResultsDashboard;
