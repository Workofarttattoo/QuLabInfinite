import { createBrowserRouter, Navigate } from 'react-router';

// Page imports
import { MissionControl } from './pages/MissionControl';
import { LiveLabWall } from './pages/LiveLabWall';
import { LiveLabWallFigma } from './pages/LiveLabWallFigma';
import { LabsByField } from './pages/LabsByField';
import { GenomicsLabVisualizer } from './pages/GenomicsLabVisualizer';
import { BoneDensityLabVisualizer } from './pages/BoneDensityLabVisualizer';
import { NeuroDiagnosticVisualizer } from './pages/NeuroDiagnosticVisualizer';
import { PharmacokineticsVisualizer } from './pages/PharmacokineticsVisualizer';
import { NeuroProductionVisualizer } from './pages/NeuroProductionVisualizer';
import { GenomicsProductionVisualizer } from './pages/GenomicsProductionVisualizer';
import { PrecisionStormVisualizer } from './pages/PrecisionStormVisualizer';
import { TacticalIntelligenceVisualizer } from './pages/TacticalIntelligenceVisualizer';
import { PharmacokineticsLabEnhanced } from './pages/PharmacokineticsLabEnhanced';
import { GlobalDashboard } from './pages/GlobalDashboard';

// Medical Labs (Ports 8001-8010)
import { AlzheimersLab } from './pages/AlzheimersLab';
import { ParkinsonsLab } from './pages/ParkinsonsLab';
import { AutoimmuneLab } from './pages/AutoimmuneLab';
import { SepsisLab } from './pages/SepsisLab';
import { WoundHealingLab } from './pages/WoundHealingLab';
import { KidneyFunctionLab } from './pages/KidneyFunctionLab';
import { LiverDiseaseLab } from './pages/LiverDiseaseLab';
import { LungFunctionLab } from './pages/LungFunctionLab';
import { PainManagementLab } from './pages/PainManagementLab';

// R&D Labs (Unified API - Port 8000)
import { MaterialsLab } from './pages/MaterialsLab';
import { QuantumLab } from './pages/QuantumLab';
import { ChemistryLab } from './pages/ChemistryLab';
import { MetabolicOptimizerLab } from './pages/MetabolicOptimizerLab';
import { CrystalInspectionLab } from './pages/CrystalInspectionLab';
import { XRDVisualization } from './pages/XRDVisualization';
import { MolecularSimulationLab } from './pages/MolecularSimulationLab';
import { CRISPREditingSuite } from './pages/CRISPREditingSuite';
import { MolecularDockingLab } from './pages/MolecularDockingLab';
import { GlobalIntelDashboard } from './pages/GlobalIntelDashboard';
import { HiveMindIntelMesh } from './pages/HiveMindIntelMesh';
import { UniversalChemistryLab } from './pages/UniversalChemistryLab';
import { XRayDiffractionLab } from './pages/XRayDiffractionLab';
import { AgentTelemetryDeepDive } from './pages/AgentTelemetryDeepDive';
import { SystemLockdownOverride } from './pages/SystemLockdownOverride';
import { SynthesisArchive } from './pages/SynthesisArchive';
import { MaterialsScienceProduction } from './pages/MaterialsScienceProduction';
import { HiveMindIntelMeshDashboard } from './pages/HiveMindIntelMeshDashboard';
import { GlobalDashboardOS } from './pages/GlobalDashboardOS';
import { MedicalDirectoryUnified } from './pages/MedicalDirectoryUnified';
import { StitchHub } from './pages/StitchHub';
import { EchoMissionFoyer } from './pages/EchoMissionFoyer';

// Echo AI Control Center
import { EchoControlCenter } from './pages/EchoControlCenter';
import { EchoWorkloadDashboard } from './pages/EchoWorkloadDashboard';
import { EchoTrainingInterface } from './pages/EchoTrainingInterface';
import { EchoSettingsPanel } from './pages/EchoSettingsPanel';
import { EchoMCPConnections } from './pages/EchoMCPConnections';

import { RootLayout } from './components/RootLayout';
import { NotFound } from './pages/NotFound';

export const router = createBrowserRouter([
  {
    path: '/',
    Component: RootLayout,
    children: [
      { index: true, Component: MissionControl },
      { path: 'dashboard', Component: GlobalDashboard },
      { path: 'dashboard-os', Component: GlobalDashboardOS },
      { path: 'medical-directory', Component: MedicalDirectoryUnified },
      { path: 'screens', Component: StitchHub },

      // Echo AI Control & Management
      { path: 'echo-mission', Component: EchoMissionFoyer },
      { path: 'echo', Component: EchoControlCenter },
      { path: 'echo/workload', Component: EchoWorkloadDashboard },
      { path: 'echo/training', Component: EchoTrainingInterface },
      { path: 'echo/settings', Component: EchoSettingsPanel },
      { path: 'echo/integrations', Component: EchoMCPConnections },

      { path: 'labs', Component: LabsByField },
      { path: 'labs-fleet', Component: LiveLabWallFigma },
      { path: 'labs-old', Component: LiveLabWall },
      { path: 'labs/alzheimers', Component: NeuroDiagnosticVisualizer },
      { path: 'labs/alzheimers-old', Component: AlzheimersLab },
      { path: 'labs/parkinsons', Component: ParkinsonsLab },
      { path: 'labs/autoimmune', Component: AutoimmuneLab },
      { path: 'labs/sepsis', Component: SepsisLab },
      { path: 'labs/wound-healing', Component: WoundHealingLab },
      { path: 'labs/bone-density', Component: BoneDensityLabVisualizer },
      { path: 'labs/kidney-function', Component: KidneyFunctionLab },
      { path: 'labs/liver-disease', Component: LiverDiseaseLab },
      { path: 'labs/lung-function', Component: LungFunctionLab },
      { path: 'labs/pain-management', Component: PainManagementLab },

      // R&D Labs (Unified API)
      { path: 'labs/materials', Component: MaterialsLab },
      { path: 'labs/xrd', Component: XRDVisualization },
      { path: 'labs/xray-diffraction', Component: XRayDiffractionLab },
      { path: 'labs/quantum', Component: QuantumLab },
      { path: 'labs/chemistry', Component: ChemistryLab },
      { path: 'labs/universal-chemistry', Component: UniversalChemistryLab },
      { path: 'labs/molecular-simulation', Component: MolecularSimulationLab },
      { path: 'labs/crystal-inspection', Component: CrystalInspectionLab },
      { path: 'labs/metabolic', Component: MetabolicOptimizerLab },
      { path: 'labs/crispr', Component: CRISPREditingSuite },
      { path: 'labs/genomics', Component: GenomicsLabVisualizer },
      { path: 'labs/drug', Component: PharmacokineticsLabEnhanced },
      { path: 'labs/drug-old', Component: PharmacokineticsVisualizer },
      { path: 'labs/molecular-docking', Component: MolecularDockingLab },
      { path: 'labs/neuro-production', Component: NeuroProductionVisualizer },
      { path: 'labs/genomics-production', Component: GenomicsProductionVisualizer },
      { path: 'labs/precision-storm', Component: PrecisionStormVisualizer },
      { path: 'labs/tactical', Component: TacticalIntelligenceVisualizer },

      // Global Dashboards
      { path: 'intel-dashboard', Component: GlobalIntelDashboard },
      { path: 'hive-mind', Component: HiveMindIntelMesh },
      { path: 'hive-mind-dashboard', Component: HiveMindIntelMeshDashboard },
      { path: 'agent-telemetry', Component: AgentTelemetryDeepDive },
      { path: 'system-lockdown', Component: SystemLockdownOverride },
      { path: 'synthesis-archive', Component: SynthesisArchive },
      { path: 'materials-production', Component: MaterialsScienceProduction },

      // Legacy Figma OS paths → real routes
      { path: 'units', element: <Navigate to="/labs" replace /> },
      { path: 'mission', element: <Navigate to="/echo-mission" replace /> },
      { path: 'system', element: <Navigate to="/system-lockdown" replace /> },
      {
        path: 'system-lockdown-override',
        element: <Navigate to="/system-lockdown" replace />,
      },

      { path: '*', Component: NotFound },
    ],
  },
]);
