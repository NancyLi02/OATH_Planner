#!/usr/bin/env python3
"""
Test script to evaluate LLM instruction parsing accuracy and generate confusion matrix.

Usage:
    # Run full evaluation
    python test_llm_instruction_parser.py
    
    # Only regenerate plots from existing predictions CSV
    python test_llm_instruction_parser.py --plot-only /path/to/predictions.csv
"""

import os
import json
import csv
import time
import argparse
from collections import defaultdict
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Intent labels
INTENT_LABELS = ["add_task", "obstacle_update", "change_task_priority"]


def get_llm_prediction(instruction: str) -> dict:
    """
    Send instruction to LLM and get predicted intent.
    Returns dict with 'intent' and 'confidence' keys.
    """
    system_prompt = """You are an intelligent assistant that classifies robot commands into specific intents.

YOUR TASK:
Analyze the user's instruction and classify it into ONE of these intents:
1. add_task - User wants to add a new task/delivery/pickup at a location
2. obstacle_update - User reports an obstacle, wall, blockage, or inaccessible area
3. change_task_priority - User wants to change the priority/urgency of an existing task

CLASSIFICATION GUIDELINES:
- "add_task": Keywords like "delivery", "pickup", "new task", "create task", "send to", "bring to", coordinates with destination
- "obstacle_update": Keywords like "wall", "obstacle", "blocked", "barrier", "avoid", "not passable", "unsafe"
- "change_task_priority": Keywords like "priority", "urgent", "first", "earlier", "important", task names with urgency

Even if the instruction is vague or ambiguous, make your best classification based on context clues.

RESPONSE FORMAT (JSON only):
{
    "intent": "add_task" | "obstacle_update" | "change_task_priority",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why you chose this intent"
}

CRITICAL: Respond with ONLY valid JSON. No markdown, no extra text.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            temperature=0.1,  # Low temperature for consistent classification
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {"intent": None, "confidence": 0.0, "reasoning": "Parse error"}
    except Exception as e:
        print(f"API error: {e}")
        return {"intent": None, "confidence": 0.0, "reasoning": str(e)}


def load_test_data(csv_path: str) -> list:
    """Load test instructions from CSV file."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['id'] and row['instruction']:  # Skip empty rows
                data.append({
                    'id': int(row['id']),
                    'instruction': row['instruction'],
                    'gt_intent': row['gt_intent'],
                    'clarity': row['clarity']
                })
    return data


def run_evaluation(test_data: list, delay: float = 0.5) -> dict:
    """
    Run evaluation on all test data.
    Returns results dict with predictions and metrics.
    """
    results = {
        'predictions': [],
        'y_true': [],
        'y_pred': [],
        'by_clarity': {'clear': [], 'ambiguous': []}
    }
    
    total = len(test_data)
    print(f"\nEvaluating {total} instructions...")
    print("=" * 60)
    
    for i, item in enumerate(test_data):
        print(f"\n[{i+1}/{total}] ID: {item['id']}")
        print(f"  Instruction: {item['instruction'][:60]}...")
        print(f"  Ground Truth: {item['gt_intent']} ({item['clarity']})")
        
        # Get LLM prediction
        prediction = get_llm_prediction(item['instruction'])
        pred_intent = prediction.get('intent')
        confidence = prediction.get('confidence', 0.0)
        
        print(f"  Predicted: {pred_intent} (confidence: {confidence:.2f})")
        
        # Check correctness
        is_correct = pred_intent == item['gt_intent']
        print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        
        # Store results
        results['predictions'].append({
            'id': item['id'],
            'instruction': item['instruction'],
            'gt_intent': item['gt_intent'],
            'pred_intent': pred_intent,
            'confidence': confidence,
            'clarity': item['clarity'],
            'correct': is_correct,
            'reasoning': prediction.get('reasoning', '')
        })
        
        results['y_true'].append(item['gt_intent'])
        results['y_pred'].append(pred_intent if pred_intent else 'unknown')
        
        # Track by clarity
        results['by_clarity'][item['clarity']].append(is_correct)
        
        # Rate limiting delay
        time.sleep(delay)
    
    return results


def compute_confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
    """Compute confusion matrix manually."""
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            i = label_to_idx[true]
            j = label_to_idx[pred]
            matrix[i, j] += 1
    
    return matrix


def compute_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """Compute precision, recall, F1 for each class and overall accuracy."""
    metrics = {}
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    metrics['accuracy'] = correct / len(y_true) if y_true else 0.0
    
    # Per-class metrics
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
    
    # Macro averages
    metrics['macro_precision'] = np.mean([metrics[l]['precision'] for l in labels])
    metrics['macro_recall'] = np.mean([metrics[l]['recall'] for l in labels])
    metrics['macro_f1'] = np.mean([metrics[l]['f1'] for l in labels])
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, labels: list, output_path: str, 
                          title: str = "LLM Intent Classification Confusion Matrix"):
    """Plot and save confusion matrix as heatmap using pure matplotlib."""
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Calculate percentages for annotations
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_percent = cm.astype('float') / row_sums * 100
    
    # Plot heatmap using imshow
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    
    # Add grid lines
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    # Add text annotations with count and percentage
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > thresh else 'black'
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=16, fontweight='bold')
    
    ax.set_xlabel('Predicted Intent', fontsize=16)
    ax.set_ylabel('True Intent', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to: {output_path}")


def plot_clarity_comparison(results: dict, output_path: str):
    """Plot accuracy comparison between clear and ambiguous instructions."""
    clear_acc = np.mean(results['by_clarity']['clear']) * 100 if results['by_clarity']['clear'] else 0
    ambig_acc = np.mean(results['by_clarity']['ambiguous']) * 100 if results['by_clarity']['ambiguous'] else 0
    
    plt.figure(figsize=(8, 6))
    
    categories = ['Clear Instructions', 'Ambiguous Instructions']
    accuracies = [clear_acc, ambig_acc]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = plt.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.ylim(0, 105)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('LLM Classification Accuracy by Instruction Clarity', fontsize=14, fontweight='bold')
    
    # Add sample sizes
    n_clear = len(results['by_clarity']['clear'])
    n_ambig = len(results['by_clarity']['ambiguous'])
    plt.xlabel(f'(n={n_clear} clear, n={n_ambig} ambiguous)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Clarity comparison saved to: {output_path}")


def generate_report(results: dict, metrics: dict, output_path: str):
    """Generate detailed text report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LLM INSTRUCTION PARSING EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Instructions: {len(results['predictions'])}\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Macro Precision: {metrics['macro_precision']*100:.2f}%\n")
        f.write(f"Macro Recall: {metrics['macro_recall']*100:.2f}%\n")
        f.write(f"Macro F1 Score: {metrics['macro_f1']*100:.2f}%\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 40 + "\n")
        for label in INTENT_LABELS:
            m = metrics[label]
            f.write(f"\n{label}:\n")
            f.write(f"  Precision: {m['precision']*100:.2f}%\n")
            f.write(f"  Recall: {m['recall']*100:.2f}%\n")
            f.write(f"  F1 Score: {m['f1']*100:.2f}%\n")
            f.write(f"  Support: {m['support']}\n")
        
        # Accuracy by clarity
        f.write("\n\nACCURACY BY INSTRUCTION CLARITY\n")
        f.write("-" * 40 + "\n")
        clear_acc = np.mean(results['by_clarity']['clear']) * 100 if results['by_clarity']['clear'] else 0
        ambig_acc = np.mean(results['by_clarity']['ambiguous']) * 100 if results['by_clarity']['ambiguous'] else 0
        f.write(f"Clear Instructions (n={len(results['by_clarity']['clear'])}): {clear_acc:.2f}%\n")
        f.write(f"Ambiguous Instructions (n={len(results['by_clarity']['ambiguous'])}): {ambig_acc:.2f}%\n")
        
        # Misclassified examples
        f.write("\n\nMISCLASSIFIED EXAMPLES\n")
        f.write("-" * 40 + "\n")
        wrong_predictions = [p for p in results['predictions'] if not p['correct']]
        
        if not wrong_predictions:
            f.write("No misclassifications!\n")
        else:
            for p in wrong_predictions:
                f.write(f"\nID {p['id']} ({p['clarity']}):\n")
                f.write(f"  Instruction: {p['instruction']}\n")
                f.write(f"  True Intent: {p['gt_intent']}\n")
                f.write(f"  Predicted: {p['pred_intent']} (conf: {p['confidence']:.2f})\n")
                f.write(f"  Reasoning: {p['reasoning']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"Report saved to: {output_path}")


def save_predictions_csv(results: dict, output_path: str):
    """Save all predictions to CSV for further analysis."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'instruction', 'gt_intent', 'pred_intent', 
            'confidence', 'clarity', 'correct', 'reasoning'
        ])
        writer.writeheader()
        writer.writerows(results['predictions'])
    
    print(f"Predictions CSV saved to: {output_path}")


def load_predictions_csv(csv_path: str) -> dict:
    """Load predictions from existing CSV file for plot-only mode."""
    results = {
        'predictions': [],
        'y_true': [],
        'y_pred': [],
        'by_clarity': {'clear': [], 'ambiguous': []}
    }
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            prediction = {
                'id': int(row['id']),
                'instruction': row['instruction'],
                'gt_intent': row['gt_intent'],
                'pred_intent': row['pred_intent'],
                'confidence': float(row['confidence']),
                'clarity': row['clarity'],
                'correct': row['correct'].lower() == 'true',
                'reasoning': row['reasoning']
            }
            results['predictions'].append(prediction)
            results['y_true'].append(row['gt_intent'])
            results['y_pred'].append(row['pred_intent'] if row['pred_intent'] else 'unknown')
            results['by_clarity'][row['clarity']].append(prediction['correct'])
    
    return results


def main():
    """Main evaluation function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Instruction Parsing Evaluation')
    parser.add_argument('--plot-only', type=str, metavar='CSV_PATH',
                        help='Only regenerate plots from existing predictions CSV file')
    args = parser.parse_args()
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from ltl_automaton_planner/ltl_automaton_planner/ to ltl_automaton_planner/
    config_dir = os.path.normpath(os.path.join(script_dir, '..', 'config'))
    
    # Output paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config_dir, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    cm_output = os.path.join(output_dir, f'confusion_matrix_{timestamp}.pdf')
    clarity_output = os.path.join(output_dir, f'clarity_comparison_{timestamp}.png')
    
    # Plot-only mode: load existing predictions and regenerate plots
    if args.plot_only:
        print("\n" + "=" * 60)
        print("PLOT-ONLY MODE")
        print("=" * 60)
        
        predictions_csv = args.plot_only
        print(f"\nLoading predictions from: {predictions_csv}")
        
        results = load_predictions_csv(predictions_csv)
        print(f"Loaded {len(results['predictions'])} predictions")
        
        # Compute metrics
        metrics = compute_metrics(results['y_true'], results['y_pred'], INTENT_LABELS)
        
        print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Macro F1 Score: {metrics['macro_f1']*100:.2f}%")
        
        # Compute and plot confusion matrix
        cm = compute_confusion_matrix(results['y_true'], results['y_pred'], INTENT_LABELS)
        print("\nConfusion Matrix:")
        print(cm)
        
        plot_confusion_matrix(cm, INTENT_LABELS, cm_output)
        plot_clarity_comparison(results, clarity_output)
        
        print("\n" + "=" * 60)
        print("PLOTS REGENERATED")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  - Confusion Matrix: {cm_output}")
        print(f"  - Clarity Comparison: {clarity_output}")
        return
    
    # Full evaluation mode
    csv_path = os.path.join(config_dir, 'llm_instruction_confusion.csv')
    report_output = os.path.join(output_dir, f'evaluation_report_{timestamp}.txt')
    predictions_output = os.path.join(output_dir, f'predictions_{timestamp}.csv')
    
    print("\n" + "=" * 60)
    print("LLM INSTRUCTION PARSING EVALUATION")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable not set!")
        return
    
    # Load test data
    print(f"\nLoading test data from: {csv_path}")
    test_data = load_test_data(csv_path)
    print(f"Loaded {len(test_data)} instructions")
    
    # Count by intent and clarity
    intent_counts = defaultdict(int)
    clarity_counts = defaultdict(int)
    for item in test_data:
        intent_counts[item['gt_intent']] += 1
        clarity_counts[item['clarity']] += 1
    
    print("\nDataset Distribution:")
    print("  By Intent:")
    for intent, count in intent_counts.items():
        print(f"    {intent}: {count}")
    print("  By Clarity:")
    for clarity, count in clarity_counts.items():
        print(f"    {clarity}: {count}")
    
    # Run evaluation
    results = run_evaluation(test_data, delay=0.3)
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)
    
    metrics = compute_metrics(results['y_true'], results['y_pred'], INTENT_LABELS)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1 Score: {metrics['macro_f1']*100:.2f}%")
    
    # Compute and plot confusion matrix
    cm = compute_confusion_matrix(results['y_true'], results['y_pred'], INTENT_LABELS)
    print("\nConfusion Matrix:")
    print(cm)
    
    plot_confusion_matrix(cm, INTENT_LABELS, cm_output)
    plot_clarity_comparison(results, clarity_output)
    
    # Generate report and save results
    generate_report(results, metrics, report_output)
    save_predictions_csv(results, predictions_output)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Confusion Matrix: {cm_output}")
    print(f"  - Clarity Comparison: {clarity_output}")
    print(f"  - Report: {report_output}")
    print(f"  - Predictions: {predictions_output}")


if __name__ == '__main__':
    main()
