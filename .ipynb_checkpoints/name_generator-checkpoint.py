import pandas as pd
import random
from typing import List, Tuple

class EmployeeNameGenerator:
    """Generate realistic employee names for the IBM dataset"""
    
    def __init__(self):
        # Lists of realistic names for professional diversity
        self.first_names_male = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
            "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
            "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
            "George", "Timothy", "Ronald", "Jason", "Edward", "Jeffrey", "Ryan", "Jacob",
            "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott",
            "Brandon", "Benjamin", "Samuel", "Gregory", "Alexander", "Patrick", "Frank",
            "Raymond", "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Henry"
        ]
        
        self.first_names_female = [
            "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
            "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Betty", "Helen", "Sandra",
            "Donna", "Carol", "Ruth", "Sharon", "Michelle", "Laura", "Sarah", "Kimberly",
            "Deborah", "Dorothy", "Lisa", "Nancy", "Karen", "Betty", "Helen", "Sandra",
            "Donna", "Carol", "Ruth", "Sharon", "Michelle", "Laura", "Emily", "Amy",
            "Angela", "Ashley", "Brenda", "Emma", "Olivia", "Cynthia", "Marie", "Janet",
            "Catherine", "Frances", "Christine", "Samantha", "Debra", "Rachel", "Carolyn"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
            "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
            "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
            "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner",
            "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
            "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper",
            "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox"
        ]
    
    def generate_employee_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic employee names to the IBM dataset"""
        
        # Create a copy to avoid modifying original
        df_with_names = df.copy()
        
        # Generate names based on gender if available, otherwise random
        employee_names = []
        used_names = set()
        
        for idx, row in df_with_names.iterrows():
            # Generate unique name
            name = self._generate_unique_name(used_names, row.get('Gender', None))
            employee_names.append(name)
            used_names.add(name)
        
        # Add names to DataFrame
        df_with_names.insert(0, 'EmployeeName', employee_names)
        
        return df_with_names
    
    def _generate_unique_name(self, used_names: set, gender: str = None) -> str:
        """Generate a unique employee name"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Select first name based on gender
            if gender == 'Male':
                first_name = random.choice(self.first_names_male)
            elif gender == 'Female':
                first_name = random.choice(self.first_names_female)
            else:
                # Random gender if not specified
                first_name = random.choice(self.first_names_male + self.first_names_female)
            
            last_name = random.choice(self.last_names)
            full_name = f"{first_name} {last_name}"
            
            if full_name not in used_names:
                return full_name
        
        # Fallback with number if all combinations exhausted
        return f"{first_name} {last_name} {random.randint(1, 999)}"
    
    def create_enhanced_sample_documents(self, df_with_names: pd.DataFrame) -> List[str]:
        """Create sample documents with actual employee names"""
        
        # Find some interesting employees to highlight
        high_performers = df_with_names[
            (df_with_names['PerformanceRating'] == 4) & 
            (df_with_names['JobSatisfaction'] >= 3)
        ].head(3)
        
        at_risk_employees = df_with_names[
            (df_with_names['Attrition'] == 'Yes')
        ].head(3)
        
        senior_employees = df_with_names[
            df_with_names['YearsAtCompany'] >= 15
        ].head(3)
        
        documents = []
        
        # High performer profiles
        for _, emp in high_performers.iterrows():
            doc = f"""
            {emp['EmployeeName']} is a high-performing employee in the {emp['Department']} department 
            working as {emp['JobRole']}. With {emp['YearsAtCompany']} years at the company and 
            a performance rating of {emp['PerformanceRating']}/4, {emp['EmployeeName']} represents 
            critical organizational knowledge in {emp['JobRole']} expertise. Monthly income of 
            ${emp['MonthlyIncome']} reflects their value to the organization. Loss of 
            {emp['EmployeeName']} would create significant knowledge gaps in {emp['Department']}.
            """
            documents.append(doc.strip())
        
        # At-risk employee analysis
        for _, emp in at_risk_employees.iterrows():
            doc = f"""
            Risk Analysis: {emp['EmployeeName']} from {emp['Department']} department shows 
            attrition risk patterns. Working as {emp['JobRole']} with {emp['YearsAtCompany']} 
            years experience, job satisfaction score of {emp['JobSatisfaction']}/4 indicates 
            potential retention challenges. Distance from home: {emp['DistanceFromHome']} miles, 
            overtime requirements: {emp['OverTime']}. Immediate retention strategies recommended 
            for {emp['EmployeeName']} to prevent knowledge loss in {emp['JobRole']} capabilities.
            """
            documents.append(doc.strip())
        
        # Senior knowledge holders
        for _, emp in senior_employees.iterrows():
            doc = f"""
            Senior Knowledge Asset: {emp['EmployeeName']} represents {emp['YearsAtCompany']} years 
            of institutional knowledge in {emp['Department']} as {emp['JobRole']}. With 
            {emp['YearsInCurrentRole']} years in current role and {emp['YearsWithCurrManager']} 
            years with current manager, {emp['EmployeeName']} possesses deep organizational 
            understanding. Education level {emp['Education']} and monthly income ${emp['MonthlyIncome']} 
            reflect senior expertise. Succession planning critical for {emp['EmployeeName']}'s 
            knowledge transfer in {emp['JobRole']} domain.
            """
            documents.append(doc.strip())
        
        return documents

# Usage function to update your dataset
def add_names_to_ibm_dataset():
    """Add names to IBM dataset and create enhanced documents"""
    try:
        # Load original IBM dataset
        df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
        print(f"Loaded IBM dataset with {len(df)} employees")
        
        # Generate names
        name_generator = EmployeeNameGenerator()
        df_with_names = name_generator.generate_employee_names(df)
        
        # Save enhanced dataset
        df_with_names.to_csv('IBM_Employee_Data_With_Names.csv', index=False)
        print(f"✅ Created enhanced dataset: IBM_Employee_Data_With_Names.csv")
        
        # Create sample documents with names
        sample_docs = name_generator.create_enhanced_sample_documents(df_with_names)
        
        # Preview some names
        print("\n📋 Sample Employee Names Generated:")
        for i in range(10):
            emp = df_with_names.iloc[i]
            print(f"• {emp['EmployeeName']} - {emp['JobRole']} ({emp['Department']})")
        
        return df_with_names, sample_docs
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    df_enhanced, sample_documents = add_names_to_ibm_dataset()
