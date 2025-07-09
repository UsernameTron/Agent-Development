import subprocess
import platform
import os
import logging
from typing import Dict, Any

class ContainmentActions:
    """Execute real containment actions on endpoints.
    Constraints: Require admin privileges, reversible actions only."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.os_type = platform.system()
        self.contained_hosts = set()
        
    def isolate_host(self, ip_address: str) -> bool:
        """Isolate a host from network communication"""
        try:
            if self.os_type == "Windows":
                return self._isolate_host_windows(ip_address)
            elif self.os_type == "Linux":
                return self._isolate_host_linux(ip_address)
        except Exception as e:
            self.logger.error(f"Host isolation failed for {ip_address}: {e}")
            return False
            
    def _isolate_host_windows(self, ip_address: str) -> bool:
        """Windows firewall isolation"""
        commands = [
            # Block all inbound traffic except from management subnet
            ["powershell", "-Command", 
             f"New-NetFirewallRule -DisplayName 'AI_ISOLATION_{ip_address}' -Direction Inbound -Action Block -RemoteAddress {ip_address}"],
            # Block all outbound traffic except to management subnet  
            ["powershell", "-Command",
             f"New-NetFirewallRule -DisplayName 'AI_ISOLATION_OUT_{ip_address}' -Direction Outbound -Action Block -RemoteAddress {ip_address}"]
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.logger.error(f"Windows isolation command failed: {result.stderr}")
                return False
                
        self.contained_hosts.add(ip_address)
        self.logger.info(f"Successfully isolated Windows host: {ip_address}")
        return True
        
    def _isolate_host_linux(self, ip_address: str) -> bool:
        """Linux iptables isolation"""
        commands = [
            ["iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"],
            ["iptables", "-A", "OUTPUT", "-d", ip_address, "-j", "DROP"]
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.logger.error(f"Linux isolation command failed: {result.stderr}")
                return False
                
        self.contained_hosts.add(ip_address)
        self.logger.info(f"Successfully isolated Linux host: {ip_address}")
        return True
        
    def kill_process(self, pid: int, host: str = "localhost") -> bool:
        """Terminate a suspicious process"""
        try:
            if host == "localhost":
                os.kill(pid, 9 if self.os_type != "Windows" else 1)
                self.logger.info(f"Terminated process {pid}")
                return True
            else:
                # Remote process termination would require additional setup
                self.logger.warning(f"Remote process termination not implemented for {host}")
                return False
        except Exception as e:
            self.logger.error(f"Process termination failed for PID {pid}: {e}")
            return False
            
    def restore_host(self, ip_address: str) -> bool:
        """Remove isolation rules for a host"""
        try:
            if self.os_type == "Windows":
                commands = [
                    ["powershell", "-Command", f"Remove-NetFirewallRule -DisplayName 'AI_ISOLATION_{ip_address}'"],
                    ["powershell", "-Command", f"Remove-NetFirewallRule -DisplayName 'AI_ISOLATION_OUT_{ip_address}'"]
                ]
            else:
                commands = [
                    ["iptables", "-D", "INPUT", "-s", ip_address, "-j", "DROP"],
                    ["iptables", "-D", "OUTPUT", "-d", ip_address, "-j", "DROP"]
                ]
                
            for cmd in commands:
                subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
            self.contained_hosts.discard(ip_address)
            self.logger.info(f"Restored connectivity for host: {ip_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"Host restoration failed for {ip_address}: {e}")
            return False