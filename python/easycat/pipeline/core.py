from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import traceback
import logging
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('easycat')

@dataclass
class DataPacket:
    """
    Container for data passed between processing nodes.
    
    Attributes
    ----------
    light_curve : Optional[pd.DataFrame]
        Main light curve data
    metadata : Dict[str, Any]
        Processing metadata and configuration
    results : Dict[str, Any]
        Results from intermediate processing steps
    errors : List[str]
        Accumulated error messages
    """
    
    light_curve: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_result(
        self,
        key: str,
        value: Any,
        node_name: Optional[str] = None
    ) -> None:
        """Store a processing result."""

        result_entry = {
            'value': value,
            'timestamp': datetime.now(),
            'node': node_name
        }

        real_key = f'{key}@{node_name}'
        self.results[real_key] = result_entry
        
    def add_error(self, message: str, node_name: str) -> None:
        """Record an error message."""

        error_entry = {
            'message': message,
            'node': node_name,
            'timestamp': datetime.now()
        }
        self.errors.append(error_entry)
    
    def add_warning(self, message: str, node_name: Optional[str] = None) -> None:
        """Record a warning message."""

        warning_entry = {
            'message': message,
            'node': node_name,
            'timestamp': datetime.now()
        }
        self.warnings.append(warning_entry)
    
    def get_result_value(self, key: str, node_name: str) -> Any:
        """Get the value of a result, ignoring metadata."""

        real_key = f'{key}@{node_name}'

        if real_key in self.results:
            return self.results[real_key]['value']
        
        return None


class NodeStatus(Enum):
    """
    Enum representing the processing state of a node.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingNode(ABC):
    """
    Each node performs a specific transformation on the data and tracks its state.
    """

    def __init__(self, name: str, minsize: int = 1):
        """
        Initialize a processing node.
        
        Parameters
        ----------
        name : str
            Unique identifier for the node
        """

        self.name = name
        self.status = NodeStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_time: Optional[float] = None
        self.config: Dict[str, Any] = {}
        self.minsize = minsize
    
    @abstractmethod
    def process(self, data: DataPacket) -> DataPacket:
        """
        Execute the node's processing logic.
        
        Parameters
        ----------
        data : DataPacket
            Input data packet
            
        Returns
        -------
        DataPacket : Modified data packet after processing
        """
        pass
    
    def validate(self, data: DataPacket) -> Tuple[bool, Optional[str]]:
        """
        Validate that input data meets node requirements.
        
        Parameters
        ----------
        data : DataPacket
            Data packet to validate
            
        Returns
        -------
        Tuple of (is_valid, error_message)
        """
        if self.minsize >= 1:
            if data.light_curve is None:
                return False, "No light curve data available"
            if len(data.light_curve) < self.minsize:
                return False, "Light curve data is empty"
        return True, None
    
    def execute(self, data: DataPacket) -> DataPacket:
        """
        Execute the node with timing and error handling.
        
        Parameters
        ----------
        data : DataPacket
            Input data packet
            
        Returns
        -------
        Processed data packet
        """

        self.start_time = datetime.now()
        self.status = NodeStatus.RUNNING
        
        logger.info(f"Starting node: {self.name}")
        try:
            # Validate input
            is_valid, error_msg = self.validate(data)

            if not is_valid:
                data.add_error(error_msg, self.name)
            else:
                data = self.process(data)
            
            # Record result
            self.end_time = datetime.now()
            self.execution_time = (self.end_time - self.start_time).total_seconds()
            if data.errors:
                self.status = NodeStatus.FAILED
                logger.info(f"Node {self.name} failed.")
            else:
                self.status = NodeStatus.COMPLETED
                logger.info(f"Node {self.name} completed in {self.execution_time:.3f}s")
        except Exception as e:
            data.add_error(traceback.format_exc(), self.name)

            self.end_time = datetime.now()
            self.execution_time = (self.end_time - self.start_time).total_seconds()
            self.status = NodeStatus.FAILED
            logger.error(f"Node {self.name} failed: {str(e)}")
            
        return data
    
    def reset(self) -> None:
        """Reset node to initial state."""
        self.status = NodeStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.execution_time = None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class Pipeline:
    """
    Manages and executes a sequence of processing nodes.
    """
    def __init__(self, name: str):
        """
        Initialize a processing pipeline.
        
        Parameters
        ----------
        name : str
            Identifier for the pipeline
        """
        self.name = name
        self.nodes: List[ProcessingNode] = []
        self.pipeline_start_time: Optional[datetime] = None
        self.pipeline_end_time: Optional[datetime] = None
        self.execution_count: int = 0
    
    def add_node(self, node: ProcessingNode) -> 'Pipeline':
        """
        Add a processing node to the pipeline.
        
        Parameters
        ----------
        node : ProcessingNode
            Processing node to add
            
        Returns
        -------
        Self for method chaining
        """
        self.nodes.append(node)
        logger.info(f"Added node '{node.name}' to pipeline '{self.name}'")
        return self
    
    def add_nodes(self, nodes: List[ProcessingNode]) -> 'Pipeline':
        for node in nodes:
            self.add_node(node)
        return self
    
    def remove_node(self, name: str) -> bool:
        """
        Remove a node from the pipeline by name.
        
        Parameters
        ----------
        name : str
            Name of the node to remove
            
        Returns
        -------
        True if node was removed, False if not found
        """
        for i, node in enumerate(self.nodes):
            if node.name == name:
                self.nodes.pop(i)
                logger.info(f"Removed node '{name}' from pipeline '{self.name}'")
                return True
        return False

    
    def run(self, data: DataPacket, break_on_error: bool = True) -> DataPacket:
        """
        Execute all nodes in the pipeline.
        
        Parameters
        ----------
        data : DataPacket
            Initial data to process
        break_on_error : bool 
            hether to stop execution if a node fails
            
        Returns
        -------
        Processed data packet
        """
        # Reset pipeline and nodes
        self.pipeline_start_time = datetime.now()
        self.execution_count += 1
        
        for node in self.nodes:
            node.reset()
        
        logger.info(f"Starting pipeline '{self.name}' with {len(self.nodes)} nodes")
        
        # Execute each node
        for i, node in enumerate(self.nodes, 1):
            node_number = f"[{i}/{len(self.nodes)}]"
            
            msg = f"{node_number} {node.name}: "
            # Execute node
            data = node.execute(data)

            if node.status == NodeStatus.FAILED:
                msg += f"✗ FAILED ({node.execution_time:.3f}s)"
            else:
                msg += f"✓ COMPLETED ({node.execution_time:.3f}s)"
            logger.info(msg+'\n')

            if break_on_error and node.status == NodeStatus.FAILED:
                logger.info(f"Pipeline stopped due to error in node: {node.name}")
                break
        
        # Record pipeline completion
        self.pipeline_end_time = datetime.now()
        
        return data
    
    def get_failed_nodes(self) -> List[str]:
        """Get names of nodes that failed in the last execution."""
        return [node.name for node in self.nodes if node.status == NodeStatus.FAILED]
