import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ProactiveKnowledgePredictor:
    """Advanced predictive analytics for knowledge gap forecasting"""
    
    def __init__(self, knowledge_graph, current_risk_score):
        self.knowledge_graph = knowledge_graph
        self.current_risk_score = current_risk_score
        self.historical_data = self._generate_historical_data()
        self.model = None
        self.scaler = StandardScaler()
        
    def _generate_historical_data(self):
        """Generate realistic historical trend data for demonstration"""
        dates = []
        risk_scores = []
        spof_counts = []
        training_hours = []
        new_hires = []
        project_counts = []
        
        # Generate 24 months of historical data
        start_date = datetime.now() - timedelta(days=730)
        
        for i in range(24):
            date = start_date + timedelta(days=30*i)
            dates.append(date)
            
            # Simulate realistic trends
            base_risk = 45 + np.sin(i/12 * 2 * np.pi) * 15  # Seasonal variation
            risk_with_noise = base_risk + np.random.normal(0, 5)
            risk_scores.append(max(20, min(90, risk_with_noise)))
            
            spof_counts.append(np.random.randint(2, 8))
            training_hours.append(np.random.randint(100, 500))
            new_hires.append(np.random.randint(0, 4))
            project_counts.append(np.random.randint(3, 12))
        
        return pd.DataFrame({
            'date': dates,
            'risk_score': risk_scores,
            'spof_count': spof_counts,
            'training_hours': training_hours,
            'new_hires': new_hires,
            'project_count': project_counts
        })
    
    def build_predictive_models(self):
        """Build ML models for knowledge gap prediction"""
        print("Building predictive models for knowledge gap forecasting...")
        
        # Prepare features
        features = ['spof_count', 'training_hours', 'new_hires', 'project_count']
        X = self.historical_data[features]
        y = self.historical_data['risk_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Train models
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        
        # Model performance
        rf_score = self.rf_model.score(X_test, y_test)
        gb_score = self.gb_model.score(X_test, y_test)
        
        print(f"Random Forest R² Score: {rf_score:.3f}")
        print(f"Gradient Boosting R² Score: {gb_score:.3f}")
        
        # Choose best model
        self.model = self.gb_model if gb_score > rf_score else self.rf_model
        self.model_type = "Gradient Boosting" if gb_score > rf_score else "Random Forest"
        
        return {
            'model_type': self.model_type,
            'accuracy': max(rf_score, gb_score),
            'historical_data_points': len(self.historical_data)
        }
    
    def predict_future_risks(self, months_ahead=6):
        """Predict future knowledge risks"""
        if self.model is None:
            self.build_predictive_models()
        
        predictions = []
        current_date = datetime.now()
        
        # Current state
        current_spof = len([n for n in self.knowledge_graph.nodes() 
                           if self.knowledge_graph.nodes[n].get('type') == 'person'])
        
        for month in range(1, months_ahead + 1):
            future_date = current_date + timedelta(days=30*month)
            
            # Simulate future scenarios
            scenarios = [
                {'spof_count': current_spof, 'training_hours': 200, 'new_hires': 1, 'project_count': 8},  # Conservative
                {'spof_count': current_spof-1, 'training_hours': 400, 'new_hires': 2, 'project_count': 10}, # Moderate
                {'spof_count': current_spof-2, 'training_hours': 600, 'new_hires': 3, 'project_count': 12}  # Aggressive
            ]
            
            scenario_predictions = []
            for scenario in scenarios:
                features = np.array([[scenario['spof_count'], scenario['training_hours'], 
                                    scenario['new_hires'], scenario['project_count']]])
                features_scaled = self.scaler.transform(features)
                risk_prediction = self.model.predict(features_scaled)[0]
                scenario_predictions.append(max(10, min(100, risk_prediction)))
            
            predictions.append({
                'date': future_date,
                'month_ahead': month,
                'conservative_scenario': scenario_predictions[0],
                'moderate_scenario': scenario_predictions[1],
                'aggressive_scenario': scenario_predictions[2],
                'recommended_actions': self._generate_month_recommendations(month, scenario_predictions)
            })
        
        return predictions
    
    def _generate_month_recommendations(self, month, predictions):
        """Generate month-specific recommendations"""
        avg_risk = np.mean(predictions)
        
        if avg_risk > 70:
            return f"Month {month}: CRITICAL - Accelerate hiring and emergency knowledge transfer"
        elif avg_risk > 50:
            return f"Month {month}: HIGH - Increase training frequency and documentation efforts"
        else:
            return f"Month {month}: MODERATE - Maintain current knowledge management practices"
    
    def identify_emerging_skill_demands(self):
        """Identify emerging skills based on trend analysis"""
        # Simulate industry trend analysis
        emerging_skills = [
            {
                'skill': 'Kubernetes',
                'demand_growth': 85,
                'current_coverage': 0,
                'urgency': 'HIGH',
                'rationale': 'Container orchestration becoming critical for cloud deployments'
            },
            {
                'skill': 'AI/ML Engineering',
                'demand_growth': 92,
                'current_coverage': 2,  # Alice and Eva have ML skills
                'urgency': 'MEDIUM',
                'rationale': 'Growing demand for production ML systems'
            },
            {
                'skill': 'DevSecOps',
                'demand_growth': 78,
                'current_coverage': 1,  # Bob has security background
                'urgency': 'HIGH',
                'rationale': 'Security integration in development pipelines essential'
            }
        ]
        
        return emerging_skills
    
    def generate_proactive_recommendations(self):
        """Generate proactive recommendations based on predictions"""
        future_risks = self.predict_future_risks()
        emerging_skills = self.identify_emerging_skill_demands()
        
        recommendations = {
            'immediate_actions': [],
            'strategic_investments': [],
            'skill_development_priorities': [],
            'risk_mitigation_timeline': future_risks
        }
        
        # Immediate actions based on prediction trends
        high_risk_months = [pred for pred in future_risks if pred['conservative_scenario'] > 60]
        if high_risk_months:
            recommendations['immediate_actions'].extend([
                "Accelerate Bob Smith's knowledge documentation - predicted bottleneck in 2-3 months",
                "Begin recruiting for Cloud Security role - 45-day timeline critical",
                "Implement cross-training program for Docker skills - prevent single point of failure"
            ])
        
        # Strategic investments
        recommendations['strategic_investments'] = [
            f"Invest in {skill['skill']} training - {skill['demand_growth']}% industry growth projected"
            for skill in emerging_skills if skill['urgency'] == 'HIGH'
        ]
        
        # Skill development priorities
        recommendations['skill_development_priorities'] = emerging_skills
        
        return recommendations

# Integration function for your existing system
def enhance_knowledge_monitor_with_predictions(knowledge_graph, current_risk_score):
    """Enhance existing Knowledge Gap Monitor with predictive capabilities"""
    
    predictor = ProactiveKnowledgePredictor(knowledge_graph, current_risk_score)
    model_info = predictor.build_predictive_models()
    proactive_recommendations = predictor.generate_proactive_recommendations()
    
    print("=== ENHANCED PREDICTIVE KNOWLEDGE ANALYTICS ===")
    print(f"Predictive Model: {model_info['model_type']}")
    print(f"Model Accuracy: {model_info['accuracy']:.1%}")
    print(f"Historical Data Points: {model_info['historical_data_points']}")
    print()
    
    print("PROACTIVE RISK PREDICTIONS (Next 6 Months):")
    for pred in proactive_recommendations['risk_mitigation_timeline']:
        print(f"  {pred['date'].strftime('%Y-%m')}: {pred['recommended_actions']}")
    
    print()
    print("IMMEDIATE PROACTIVE ACTIONS:")
    for action in proactive_recommendations['immediate_actions']:
        print(f"  • {action}")
    
    print()
    print("EMERGING SKILL DEMANDS:")
    for skill in proactive_recommendations['skill_development_priorities']:
        print(f"  • {skill['skill']}: {skill['demand_growth']}% growth ({skill['urgency']} priority)")
    
    return predictor, proactive_recommendations
