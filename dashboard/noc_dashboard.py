#!/usr/bin/env python3
"""Simple NOC dashboard using textual/rich for real-time monitoring"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
except ImportError:
    print("Install rich: pip install rich")
    exit(1)

class NOCDashboard:
    def __init__(self):
        self.console = Console()
        self.metrics_file = Path("logs/agent_metrics.json")
        
    def load_metrics(self):
        """Load latest agent metrics"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            return {"error": str(e)}
        return {}
        
    def create_agent_status_table(self, metrics):
        """Create table showing agent status"""
        table = Table(title="NOC Agent Status", show_header=True, header_style="bold magenta")
        table.add_column("Agent ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Detections", style="blue")
        table.add_column("Fitness", style="red")
        
        agents = metrics.get("agents", {})
        for agent_id, agent_data in agents.items():
            status = "🟢 Active" if agent_data.get("active", False) else "🔴 Inactive"
            detections = agent_data.get("security_detections", {})
            tp = detections.get("true_positives", 0)
            fp = detections.get("false_positives", 0)
            fitness = f"{agent_data.get('fitness_security', 0):.2f}"
            
            table.add_row(
                agent_id[:8],
                agent_data.get("type", "Unknown"),
                status,
                f"TP:{tp} FP:{fp}",
                fitness
            )
            
        return table
        
    def create_threat_summary(self, metrics):
        """Create threat detection summary"""
        threats = metrics.get("recent_threats", [])
        
        if not threats:
            return Panel("[green]No recent threats detected[/green]", title="Threat Summary")
            
        threat_text = ""
        for threat in threats[-5:]:  # Last 5 threats
            severity = threat.get("severity", "LOW")
            color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}.get(severity, "white")
            threat_text += f"[{color}]{threat.get('timestamp', '')} - {threat.get('type', 'Unknown')} ({severity})[/{color}]\n"
            
        return Panel(threat_text, title="Recent Threats")
        
    def create_performance_panel(self, metrics):
        """Create performance metrics panel"""
        perf = metrics.get("performance", {})
        
        detection_rate = perf.get("detection_rate", 0)
        false_positive_rate = perf.get("false_positive_rate", 0)
        avg_response_time = perf.get("avg_response_time", 0)
        
        perf_text = f"""
Detection Rate: {detection_rate:.1%}
False Positive Rate: {false_positive_rate:.1%}
Avg Response Time: {avg_response_time:.1f}s
Active Specialists: {perf.get("active_specialists", 0)}
        """
        
        return Panel(perf_text, title="Performance Metrics")
        
    def create_layout(self, metrics):
        """Create dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["header"].update(
            Panel(f"🛡️ NOC AI Agents Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                  style="bold blue")
        )
        
        layout["body"].split_row(
            Layout(self.create_agent_status_table(metrics), name="agents"),
            Layout(name="side")
        )
        
        layout["side"].split_column(
            Layout(self.create_threat_summary(metrics), name="threats"),
            Layout(self.create_performance_panel(metrics), name="performance")
        )
        
        layout["footer"].update(
            Panel("[bold green]Press Ctrl+C to exit[/bold green]", style="dim")
        )
        
        return layout
        
    def run(self):
        """Run the dashboard"""
        with Live(self.create_layout({}), refresh_per_second=2, screen=True) as live:
            try:
                while True:
                    metrics = self.load_metrics()
                    live.update(self.create_layout(metrics))
                    time.sleep(1)
            except KeyboardInterrupt:
                self.console.print("\n[bold red]Dashboard stopped[/bold red]")

if __name__ == "__main__":
    dashboard = NOCDashboard()
    dashboard.run()