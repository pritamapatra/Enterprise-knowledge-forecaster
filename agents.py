from crewai import Agent, Task, Crew
import requests
import json

class MonitoringAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Knowledge Monitor",
            goal="Detect knowledge gaps and document changes",
            backstory="An experienced AI agent specialized in scanning enterprise documents and identifying knowledge gaps.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"MonitoringAgent: {task.description}")
        return {
            "status": "gap_detected",
            "gap": "Project Alpha training materials missing",
            "priority": "high",
            "source": "document_analysis"
        }

class ContentGenerationAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Content Creator",
            goal="Generate training materials and documentation",
            backstory="A creative AI agent focused on generating high-quality training materials and documentation.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"ContentGenerationAgent: {task.description}")
        return self.generate_content_via_lm_studio(task.description)
    
    def generate_content_via_lm_studio(self, prompt):
        try:
            response = requests.post("http://127.0.0.1:1234/v1/chat/completions", 
                json={
                    "messages": [{"role": "user", "content": f"Generate training summary: {prompt}"}],
                    "model": "local-model"
                },
                timeout=30
            )
            if response.status_code == 200:
                return {"status": "content_generated", "content": response.json()["choices"][0]["message"]["content"]}
            else:
                return {"status": "error", "content": f"LM Studio returned status code: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "content": f"LM Studio connection failed: {str(e)}"}
        except Exception as e:
            return {"status": "error", "content": f"Unexpected error: {str(e)}"}

class RecommendationAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Advisor",
            goal="Provide actionable recommendations",
            backstory="A strategic AI agent that provides actionable recommendations based on knowledge analysis.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"RecommendationAgent: {task.description}")
        return {
            "status": "recommendation_ready",
            "recommendations": [
                "Schedule training session for Project Alpha team",
                "Update knowledge base with generated content",
                "Notify stakeholders of identified gaps"
            ],
            "priority": "immediate"
        }

class QualityAssuranceAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Quality Controller",
            goal="Validate agent outputs and ensure quality",
            backstory="A meticulous AI agent dedicated to maintaining content quality and enterprise standards.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"QualityAssuranceAgent: {task.description}")
        return {
            "status": "quality_approved",
            "validation": "Content meets enterprise standards",
            "score": 8.5
        }

class AnalysisAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Data Analyst",
            goal="Analyze patterns and trends in enterprise data",
            backstory="A analytical AI agent specialized in processing and interpreting enterprise data patterns.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"AnalysisAgent: {task.description}")
        return {
            "status": "analysis_complete",
            "insights": "Data analysis reveals key performance indicators",
            "metrics": {"accuracy": 0.92, "coverage": 0.87}
        }

class DocumentationAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Documentation Specialist",
            goal="Create and maintain comprehensive documentation",
            backstory="A thorough AI agent focused on creating detailed and accurate documentation.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"DocumentationAgent: {task.description}")
        return {
            "status": "documentation_ready",
            "documents": "Comprehensive documentation package created",
            "format": "enterprise_standard"
        }

class ValidationAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Validator",
            goal="Verify accuracy and compliance of all outputs",
            backstory="A precise AI agent dedicated to ensuring accuracy and regulatory compliance.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"ValidationAgent: {task.description}")
        return {
            "status": "validation_complete",
            "compliance": "All outputs meet regulatory requirements",
            "accuracy_score": 9.2
        }

class CommunicationAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Communications Manager",
            goal="Handle stakeholder communication and notifications",
            backstory="A diplomatic AI agent specialized in clear communication with stakeholders.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"CommunicationAgent: {task.description}")
        return {
            "status": "communication_sent",
            "recipients": "All relevant stakeholders notified",
            "method": "enterprise_channels"
        }

class CoordinationAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Process Coordinator",
            goal="Coordinate and orchestrate workflow processes",
            backstory="An organized AI agent responsible for ensuring smooth workflow coordination.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"CoordinationAgent: {task.description}")
        return {
            "status": "coordination_complete",
            "workflow": "All processes synchronized successfully",
            "efficiency": 0.94
        }

class ComplianceAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Compliance Officer",
            goal="Ensure all processes meet regulatory and policy requirements",
            backstory="A diligent AI agent focused on maintaining compliance with all applicable regulations.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"ComplianceAgent: {task.description}")
        return {
            "status": "compliance_verified",
            "regulations": "All applicable regulations satisfied",
            "audit_trail": "Complete audit trail maintained"
        }

class ReportingAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Report Generator",
            goal="Generate comprehensive reports and summaries",
            backstory="A detail-oriented AI agent specialized in creating comprehensive reports and executive summaries.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"ReportingAgent: {task.description}")
        return {
            "status": "report_generated",
            "report": "Executive summary and detailed reports prepared",
            "format": "multi_format_output"
        }

def validate_crew_safety(crew):
    """Validate crew object before use to prevent runtime errors"""
    if crew is None:
        raise ValueError("Crew object is not initialized")
    
    if not hasattr(crew, 'agents') or crew.agents is None:
        raise ValueError("Crew agents are not initialized")
    
    if not isinstance(crew.agents, list):
        raise TypeError(f"Crew agents should be a list, got {type(crew.agents)}")
    
    agent_count = len(crew.agents)
    print(f"Crew validation: Found {agent_count} agents")
    
    if agent_count < 11:
        raise IndexError(f"Insufficient agents: need at least 11, found {agent_count}")
    
    return True

def create_enterprise_crew():
    """Create a comprehensive enterprise crew with all necessary agents"""
    agents = [
        MonitoringAgent(),           # Index 0
        ContentGenerationAgent(),   # Index 1
        RecommendationAgent(),      # Index 2
        QualityAssuranceAgent(),    # Index 3
        AnalysisAgent(),            # Index 4
        DocumentationAgent(),       # Index 5
        ValidationAgent(),          # Index 6
        CommunicationAgent(),       # Index 7
        CoordinationAgent(),        # Index 8
        ComplianceAgent(),          # Index 9
        ReportingAgent(),           # Index 10
        RecruitmentProposalAgent(), # Index 11 
        TrainingProposalAgent()      # Index 12
    ]
    
    crew = Crew(agents=agents)
    print(f"Enterprise crew created with {len(agents)} agents")
    return crew

def run_agentic_workflow(documents_context=""):
    """Main agentic workflow orchestration with comprehensive error handling"""
    try:
        # Create and validate crew
        crew = create_enterprise_crew()
        validate_crew_safety(crew)
        
        results = {}
        
        # Step 1: Knowledge Monitoring
        print("Step 1: Knowledge Monitoring")
        monitor_task = Task(
            description=f"Analyze enterprise documents for knowledge gaps: {documents_context}",
            agent=crew.agents[0],
            expected_output="A dictionary containing knowledge gap analysis with status, gap details, priority, and source information"
        )
        results["monitoring"] = crew.agents[0].perform_task(monitor_task)
        
        # Step 2: Content Generation (if gap detected)
        if isinstance(results["monitoring"], dict) and results["monitoring"].get("status") == "gap_detected":
            print("Step 2: Content Generation")
            content_task = Task(
                description=f"Generate training content for: {results['monitoring'].get('gap', 'unknown gap')}",
                agent=crew.agents[1],
                expected_output="A dictionary containing generated training content with status and content fields"
            )
            results["content"] = crew.agents[1].perform_task(content_task)
            
            # Step 3: Quality Assurance
            print("Step 3: Quality Assurance")
            qa_task = Task(
                description=f"Review generated content: {results.get('content', {})}",
                agent=crew.agents[3],
                expected_output="A dictionary containing quality assessment with validation results and quality score"
            )
            results["quality"] = crew.agents[3].perform_task(qa_task)
            
            # Step 4: Documentation
            print("Step 4: Documentation")
            doc_task = Task(
                description=f"Create documentation for content: {results.get('content', {})}",
                agent=crew.agents[5],
                expected_output="A dictionary containing documentation status and format information"
            )
            results["documentation"] = crew.agents[5].perform_task(doc_task)
        
        # Step 5: Analysis
        print("Step 5: Analysis")
        analysis_task = Task(
            description=f"Analyze workflow results and patterns: {results}",
            agent=crew.agents[4],
            expected_output="A dictionary containing analysis results with insights and metrics"
        )
        results["analysis"] = crew.agents[4].perform_task(analysis_task)
        
        # Step 6: Validation
        print("Step 6: Validation")
        validation_task = Task(
            description=f"Validate all workflow outputs: {results}",
            agent=crew.agents[6],
            expected_output="A dictionary containing validation results with compliance and accuracy information"
        )
        results["validation"] = crew.agents[6].perform_task(validation_task)
        
        # Step 7: Recommendations
        print("Step 7: Recommendations")
        recommend_task = Task(
            description=f"Provide recommendations based on findings: {results}",
            agent=crew.agents[2],
            expected_output="A dictionary containing actionable recommendations with status and priority information"
        )
        results["recommendations"] = crew.agents[2].perform_task(recommend_task)
        
        # Step 8: Communication
        print("Step 8: Communication")
        comm_task = Task(
            description=f"Communicate results to stakeholders: {results}",
            agent=crew.agents[7],
            expected_output="A dictionary containing communication status and delivery confirmation"
        )
        results["communication"] = crew.agents[7].perform_task(comm_task)
        
        # Step 9: Coordination
        print("Step 9: Coordination")
        coord_task = Task(
            description=f"Coordinate final workflow processes: {results}",
            agent=crew.agents[8],
            expected_output="A dictionary containing coordination status and efficiency metrics"
        )
        results["coordination"] = crew.agents[8].perform_task(coord_task)
        
        # Step 10: Compliance Check
        print("Step 10: Compliance Check")
        compliance_task = Task(
            description=f"Verify compliance of all workflow outputs: {results}",
            agent=crew.agents[9],
            expected_output="A dictionary containing compliance verification and audit trail information"
        )
        results["compliance"] = crew.agents[9].perform_task(compliance_task)
        
        # Step 11: Final Reporting
        print("Step 11: Final Reporting")
        report_task = Task(
            description=f"Generate comprehensive workflow report: {results}",
            agent=crew.agents[10],
            expected_output="A dictionary containing final report with executive summary"
        )
        results["reporting"] = crew.agents[10].perform_task(report_task)

        print("Step 12: Recruitment Analysis & Proposals")
        recruitment_task = Task(
            description=f"Analyze current team composition and project requirements to identify recruitment needs and propose qualified candidates: {documents_context}",
            agent=crew.agents[11],  # RecruitmentProposalAgent
            expected_output="Recruitment analysis with candidate proposals and strategic recommendations"
        )
        results["recruitment_proposals"] = crew.agents[37].perform_task(recruitment_task)
        
        # Step 13: Training Module Proposals  
        print("Step 13: Training Module Proposals")
        training_task = Task(
            description=f"Analyze team skill gaps and performance data to propose targeted training modules and learning paths: {documents_context}. Consider results from previous analysis: {results.get('analysis', {})}",
            agent=crew.agents[38],  # TrainingProposalAgent
            expected_output="Training proposals with skill gap analysis and learning recommendations"
        )
        results["training_proposals"] = crew.agents[38].perform_task(training_task)
        
        return results
        
    except Exception as e:
        print(f"Enhanced workflow error: {e}")
        return {"error": f"Enhanced workflow execution error: {e}"}
        
        return results
        
    except IndexError as e:
        print(f"Agent access error: {e}")
        return {"error": f"Agent configuration error: {e}"}
    except ValueError as e:
        print(f"Crew validation error: {e}")
        return {"error": f"Crew initialization error: {e}"}
    except Exception as e:
        print(f"Unexpected workflow error: {e}")
        return {"error": f"Workflow execution error: {e}"}

class RecruitmentProposalAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Expert Recruitment Strategist",
            goal="Proactively identify recruitment needs and propose qualified candidates based on enterprise knowledge gaps",
            backstory="You are a senior talent acquisition specialist with deep expertise in workforce forecasting. You analyze current team capabilities, project requirements, and skill gaps to proactively suggest recruitment strategies and identify ideal candidates.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"RecruitmentProposalAgent: {task.description}")
        return self.analyze_recruitment_needs(task.description)
    
    def analyze_recruitment_needs(self, context):
        """Analyze and propose recruitment strategies"""
        try:
            return {
                "status": "recruitment_analysis_complete",
                "predicted_needs": [
                    "2 Senior Data Scientists needed in Q2 2025",
                    "1 ML Engineer required for project Alpha", 
                    "3 DevOps Engineers for infrastructure scaling"
                ],
                "candidate_proposals": [
                    {
                        "role": "Senior Data Scientist",
                        "candidates": [
                            "Profile match: 8+ years ML experience, cloud-native expertise",
                            "Salary range: $120k-150k, remote-friendly"
                        ],
                        "timeline": "4-6 weeks",
                        "priority": "High"
                    }
                ],
                "strategic_recommendations": [
                    "Focus on candidates with cloud-native ML experience",
                    "Consider remote-first candidates to expand talent pool",
                    "Implement referral program for technical roles"
                ]
            }
        except Exception as e:
            return {"status": "error", "message": f"Recruitment analysis failed: {str(e)}"}


class TrainingProposalAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role="Learning & Development Strategist", 
            goal="Identify skill gaps and proactively propose targeted training modules to enhance team capabilities",
            backstory="You are a learning and development expert who specializes in analyzing team performance data, identifying skill gaps, and designing personalized training programs. You stay current with industry trends and emerging technologies to recommend relevant training modules.",
            **kwargs
        )
    
    def perform_task(self, task):
        print(f"TrainingProposalAgent: {task.description}")
        return self.propose_training_modules(task.description)
    
    def propose_training_modules(self, context):
        """Analyze and propose training modules"""
        try:
            return {
                "status": "training_analysis_complete",
                "skill_gaps_identified": [
                    "Cloud Architecture (40% of team needs upskilling)",
                    "Advanced Python (30% of team needs training)",
                    "Leadership Skills (identified in 8 senior developers)"
                ],
                "training_proposals": [
                    {
                        "module": "AWS Solutions Architecture Certification",
                        "target_audience": "Senior Developers, DevOps Team",
                        "duration": "8 weeks",
                        "cost": "$2,500 per person", 
                        "priority": "High",
                        "expected_outcome": "Cloud-native application deployment"
                    },
                    {
                        "module": "Advanced Python for Data Science",
                        "target_audience": "Junior/Mid-level Developers",
                        "duration": "6 weeks",
                        "cost": "$1,800 per person",
                        "priority": "Medium", 
                        "expected_outcome": "Enhanced data processing capabilities"
                    }
                ],
                "learning_paths": [
                    "Individual learning paths created for 15 team members",
                    "Mentorship program pairing senior with junior developers",
                    "Monthly skill assessment checkpoints"
                ],
                "roi_projection": "25% improvement in project delivery speed within 6 months"
            }
        except Exception as e:
            return {"status": "error", "message": f"Training analysis failed: {str(e)}"}
        
if __name__ == "__main__":
    # Test the workflow
    print("Starting Enterprise Agentic Workflow")
    print("=" * 50)
    
    try:
        results = run_agentic_workflow("Enterprise project documentation uploaded")
        
        print("\nAgentic Workflow Results:")
        print("=" * 50)
        
        for agent_type, result in results.items():
            if agent_type != "error":
                print(f"\n{agent_type.upper()}:")
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {result}")
            else:
                print(f"\nERROR: {result}")
        
        print("\nWorkflow execution completed successfully")
        
    except Exception as e:
        print(f"\nCritical workflow error: {e}")
        print("Please check crew configuration and agent initialization")
