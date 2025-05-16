#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import List, Dict, Any, Optional, Union

from fuzzer.utils.utils import initialize_logger
from .population import Population, log_population_info
from .rag_enhanced_generator import RAGEnhancedGenerator

class RAGEnhancedPopulation(Population):
    """
    Quần thể được tăng cường bởi RAG, sử dụng phân tích dataflow để tạo các chuỗi giao dịch thông minh hơn
    """
    
    def __init__(self, indv_template, indv_generator, size=100, other_generators=None):
        super().__init__(indv_template, indv_generator, size, other_generators)
        self.logger = initialize_logger("RAGEnhancedPopulation")
        
    def init(self, indvs=None, init_seed=False, no_cross=False):
        """
        Khởi tạo quần thể với các chuỗi transaction được sinh bởi RAG + dataflow
        """
        IndvType = self.indv_template.__class__
        
        # Nếu generator không phải RAGEnhancedGenerator, sử dụng phương thức mặc định
        if not isinstance(self.indv_generator, RAGEnhancedGenerator):
            self.logger.info("Using standard population initialization (non-RAG generator)")
            return super().init(indvs, init_seed, no_cross)
        
        # Ghi log thông tin về quá trình khởi tạo
        self.logger.info("Initializing RAG-enhanced population")
        
        if indvs is None:
            # Lấy các sequence tối ưu từ RAG
            optimal_sequences = self.indv_generator.optimal_sequences
            vulnerabilities = self.indv_generator.potential_vulnerabilities
            critical_paths = self.indv_generator.critical_paths
            
            self.logger.info(f"Analysis data: {len(optimal_sequences)} optimal sequences, {len(vulnerabilities)} vulnerabilities, {len(critical_paths)} critical paths")
            
            # Phân bổ 70% quần thể cho các sequence thông minh, 30% cho các sequence ngẫu nhiên
            smart_size = int(0.7 * self.size)
            random_size = self.size - smart_size
            
            # Theo dõi số lượng cá thể đã tạo
            created_individuals = 0
            
            # Tạo các cá thể từ sequence tối ưu
            if optimal_sequences:
                self.logger.info("Creating individuals from optimal sequences")
                for sequence_template in optimal_sequences[:min(len(optimal_sequences), smart_size // 3)]:
                    # Tạo cá thể từ template
                    chromosome = self.indv_generator._generate_optimal_sequence(sequence_template)
                    if chromosome:  # Kiểm tra nếu có transaction hợp lệ
                        indv = IndvType(generator=self.indv_generator, 
                                       other_generators=self.indv_generator.other_generators).init(chromosome=chromosome)
                        self.individuals.append(indv)
                        created_individuals += 1
            
            # Tạo các cá thể từ lỗ hổng tiềm ẩn
            if vulnerabilities and created_individuals < smart_size:
                self.logger.info("Creating individuals targeting vulnerabilities")
                for vulnerability in vulnerabilities[:min(len(vulnerabilities), (smart_size - created_individuals) // 2)]:
                    # Tạo cá thể nhắm vào lỗ hổng
                    chromosome = self.indv_generator._generate_vulnerability_targeting_sequence(vulnerability)
                    if chromosome:
                        indv = IndvType(generator=self.indv_generator, 
                                       other_generators=self.indv_generator.other_generators).init(chromosome=chromosome)
                        self.individuals.append(indv)
                        created_individuals += 1
            
            # Tạo các cá thể từ critical paths
            if critical_paths and created_individuals < smart_size:
                self.logger.info("Creating individuals from critical paths")
                for path in critical_paths[:min(len(critical_paths), smart_size - created_individuals)]:
                    if path and len(path) > 0:
                        # Tạo cá thể từ critical path
                        chromosome = self.indv_generator._generate_related_functions_sequence(path[0])
                        if chromosome:
                            indv = IndvType(generator=self.indv_generator, 
                                          other_generators=self.indv_generator.other_generators).init(chromosome=chromosome)
                            self.individuals.append(indv)
                            created_individuals += 1
            
            # Bổ sung thêm các cá thể ngẫu nhiên nếu cần
            self.logger.info(f"Adding {self.size - created_individuals} random individuals to reach population size {self.size}")
            while len(self.individuals) < self.size:
                # Tạo cá thể ngẫu nhiên
                indv = IndvType(generator=self.indv_generator, 
                               other_generators=self.indv_generator.other_generators).init(no_cross=no_cross)
                self.individuals.append(indv)
        else:
            # Sử dụng các cá thể đã được cung cấp
            self.logger.info(f"Using {len(indvs)} provided individuals")
            self.individuals = indvs
        
        self._updated = True
        self.size = len(self.individuals)
        
        # Log thông tin về quần thể
        self.logger.info(f"Population initialized with {self.size} individuals")
        log_population_info(self.individuals)
        
        return self 