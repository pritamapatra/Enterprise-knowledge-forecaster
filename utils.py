import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.strip()

def extract_financial_figures(text: str) -> Dict[str, str]:
    """Extract financial figures from text"""
    financial_patterns = {
        'millions': r'\$(\d+(?:\.\d+)?)\s*[Mm](?:illion)?',
        'thousands': r'\$(\d+(?:,\d{3})*(?:\.\d+)?)',
        'percentages': r'(\d+(?:\.\d+)?)\%',
        'roi': r'(?:ROI|return on investment).*?(\d+(?:\.\d+)?)\%'
    }
    
    results = {}
    for key, pattern in financial_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            results[key] = matches
    
    return results

def format_currency(amount: float) -> str:
    """Format currency values"""
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.0f}K"
    else:
        return f"${amount:.0f}"

def extract_key_metrics(text: str) -> Dict[str, Any]:
    """Extract key business metrics from text"""
    metrics = {}
    
    # Investment amounts
    investment_match = re.search(r'\$(\d+(?:\.\d+)?)\s*million.*investment', text, re.IGNORECASE)
    if investment_match:
        metrics['investment'] = float(investment_match.group(1))
    
    # ROI percentages
    roi_match = re.search(r'(?:ROI|return).*?(\d+(?:\.\d+)?)\%', text, re.IGNORECASE)
    if roi_match:
        metrics['roi'] = float(roi_match.group(1))
    
    # Employee counts
    employee_match = re.search(r'(\d+(?:,\d{3})*)\s+employees', text, re.IGNORECASE)
    if employee_match:
        metrics['employees'] = int(employee_match.group(1).replace(',', ''))
    
    return metrics
