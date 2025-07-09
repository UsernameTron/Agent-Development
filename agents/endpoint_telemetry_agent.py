import json
import subprocess
import platform
from datetime import datetime
from agents.base_agent import BaseAgent

class EndpointTelemetryAgent(BaseAgent):
    """Collects endpoint security telemetry for baseline learning.
    Constraints: Read-only operations, 30s intervals max."""
    
    def __init__(self, name="EndpointTelemetryAgent"):
        super().__init__(name)
        self.os_type = platform.system()
        self.knowledge_base["telemetry_raw"] = []
        self.knowledge_base["baseline_stats"] = {}
        
    def collect_windows_events(self):
        """Collect Windows Security Event Log entries"""
        try:
            # PowerShell command to get recent security events
            cmd = [
                "powershell", "-Command",
                "Get-WinEvent -FilterHashtable @{LogName='Security'; ID=4624,4625,4648,4672} -MaxEvents 50 | ConvertTo-Json"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                events = json.loads(result.stdout)
                if not isinstance(events, list):
                    events = [events]
                    
                for event in events:
                    telemetry_event = {
                        "timestamp": datetime.now().isoformat(),
                        "event_id": event.get("Id"),
                        "level": event.get("LevelDisplayName"),
                        "source": "Windows Security Log",
                        "machine": event.get("MachineName"),
                        "user": event.get("UserId"),
                        "raw_data": event
                    }
                    self.knowledge_base["telemetry_raw"].append(telemetry_event)
                    
        except Exception as e:
            self.logger.error(f"Windows event collection failed: {e}")
            
    def collect_linux_logs(self):
        """Collect Linux system logs via journalctl"""
        try:
            cmd = ["journalctl", "-n", "50", "--output=json", "--since", "5 minutes ago"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        log_entry = json.loads(line)
                        telemetry_event = {
                            "timestamp": datetime.now().isoformat(),
                            "event_id": log_entry.get("PRIORITY"),
                            "level": log_entry.get("PRIORITY"),
                            "source": "Linux Journal",
                            "machine": log_entry.get("_HOSTNAME"),
                            "message": log_entry.get("MESSAGE"),
                            "raw_data": log_entry
                        }
                        self.knowledge_base["telemetry_raw"].append(telemetry_event)
                        
        except Exception as e:
            self.logger.error(f"Linux log collection failed: {e}")
    
    def run(self, task):
        """Main collection routine - called every 30 seconds"""
        self.logger.info("Collecting endpoint telemetry...")
        
        if self.os_type == "Windows":
            self.collect_windows_events()
        elif self.os_type == "Linux":
            self.collect_linux_logs()
            
        # Trim old data to prevent memory bloat
        if len(self.knowledge_base["telemetry_raw"]) > 1000:
            self.knowledge_base["telemetry_raw"] = self.knowledge_base["telemetry_raw"][-500:]
            
        return f"Collected telemetry: {len(self.knowledge_base['telemetry_raw'])} events"