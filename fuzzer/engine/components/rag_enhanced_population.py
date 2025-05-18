#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import json
import logging
from typing import Dict, List, Any, Optional

from fuzzer.utils.utils import initialize_logger
from engine.components.population import Population

class RAGEnhancedPopulation(Population):
    """
    Enhanced Population sử dụng RAG để tối ưu việc sinh dữ liệu
    """
    
    def __init__(self, indv_template, indv_generator, size=10, other_generators=None):
        """
        Khởi tạo RAG Enhanced Population
        """
        super().__init__(indv_template, indv_generator, size, other_generators)
        self.logger = initialize_logger("RAGEnhancedPopulation")
        self.logger.info("Initialized RAGEnhancedPopulation")
    
    def init(self, init_seed=True):
        """
        Initialize population with individuals.
        """
        self.logger.info(f"Initializing population with {self.size} individuals")
        for i in range(self.size):
            indv = self.indv_template.clone()
            self.append(indv)

            if init_seed:
                self[-1].seed()
                
        return self 