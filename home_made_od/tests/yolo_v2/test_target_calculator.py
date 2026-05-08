import torch
import numpy as np
import pytest
from object_detection_mania.yolo_v2.modules.target_calculation import YoloV2TargetCalculator

class TestYoloV2TargetCalculatorStressTesting:
    """
    Stress tests the objectness and class components of the target elements
    using diverse, deterministic geometric strategies to guarantee anchor assignment.
    """

    @pytest.fixture
    def setup_params(self):
        return {
            "num_iterations": 100,  # High iteration count for stress testing
            "max_batch_size": 6,
            "max_num_classes": 15,
            "h_range": (5, 13),
            "w_range": (5, 13),
            "num_anchors": 5,       # Fixed to 5 for standard YOLOv2 testing
        }

    def _run_exhaustive_test(self, setup_params, anchor_generator, target_generator):
        """
        A DRY helper method that runs the exhaustive randomized loop using
        injected generator logic for anchors and bounding boxes/targets.
        """
        for _ in range(setup_params["num_iterations"]):
            batch_size = np.random.randint(1, setup_params["max_batch_size"])
            num_classes = np.random.randint(2, setup_params["max_num_classes"])
            h = np.random.randint(*setup_params["h_range"])
            w = np.random.randint(*setup_params["w_range"])
            num_anchors = setup_params["num_anchors"]

            # Generate geometrically distinct anchors based on strategy
            anchors = anchor_generator(num_anchors)
            calculator = YoloV2TargetCalculator(num_classes, anchors, (h, w))

            max_cells = max(1, (h * w) // 2)
            N = np.random.randint(1, max_cells + 1)
            all_gt = []
            expected_assignments = {}

            for b in range(batch_size):
                chosen_cells = np.random.choice(h * w, N, replace=False)
                for cell_idx in chosen_cells:
                    row = cell_idx // w
                    col = cell_idx % w
                    
                    M = np.random.randint(1, num_anchors + 1)
                    target_anchor_indices = np.random.choice(num_anchors, M, replace=False)
                    
                    for anchor_idx in target_anchor_indices:
                        cls_id = np.random.randint(0, num_classes)
                        
                        # Use inverse logic: generate the expected targets, then create the GT box
                        cx, cy, bw, bh, expected_tx, expected_ty, expected_tw, expected_th = target_generator(anchor_idx, anchors, row, col, h, w)
                        
                        # Clip to valid ranges to prevent anomalous bounding boxes
                        bw = np.clip(bw, 1e-4, 1.0)
                        bh = np.clip(bh, 1e-4, 1.0)
                        
                        all_gt.append([float(b), float(cls_id), float(cx), float(cy), float(bw), float(bh)])
                        expected_assignments[(b, anchor_idx, row, col)] = {
                            "cls": cls_id,
                            "tx": expected_tx,
                            "ty": expected_ty,
                            "tw": expected_tw,
                            "th": expected_th
                        }
                        
            targets = torch.tensor(all_gt, dtype=torch.float32)
            
            # Compute
            target_tensor = calculator.compute_targets(targets, batch_size)
            reshaped_targets = target_tensor.view(batch_size, num_anchors, 5 + num_classes, h, w)
            
            # Assertions
            expected_total_objects = len(expected_assignments)
            assert reshaped_targets[:, :, 4, :, :].sum().item() == expected_total_objects
            assert reshaped_targets[:, :, 5:, :, :].sum().item() == expected_total_objects
            
            for (b, a, row, col), expected_data in expected_assignments.items():
                expected_cls = expected_data["cls"]
                
                # Verify Objectness & Class
                assert reshaped_targets[b, a, 4, row, col] == 1.0
                assert reshaped_targets[b, a, 5 + expected_cls, row, col] == 1.0
                assert reshaped_targets[b, a, 5:, row, col].sum().item() == 1.0
                
                # Verify Regression Targets
                actual_tx = reshaped_targets[b, a, 0, row, col].item()
                actual_ty = reshaped_targets[b, a, 1, row, col].item()
                actual_tw = reshaped_targets[b, a, 2, row, col].item()
                actual_th = reshaped_targets[b, a, 3, row, col].item()
                
                assert np.isclose(actual_tx, expected_data["tx"], atol=1e-6), f"tx mismatch at {b},{a},{row},{col}: {actual_tx} != {expected_data['tx']}"
                assert np.isclose(actual_ty, expected_data["ty"], atol=1e-6), f"ty mismatch at {b},{a},{row},{col}: {actual_ty} != {expected_data['ty']}"
                
                # For tw and th, the calculator adds 1e-7 to the denominator and numerator to avoid log(0).
                # When we generate inverse targets, the math is pure. The epsilon causes a slight deviation.
                # Therefore we apply the same epsilon to our expected calculation or use a slightly looser tolerance.
                # The user requested 1e-6, which is usually sufficient if the epsilon effect is small.
                assert np.isclose(actual_tw, expected_data["tw"], atol=1e-6), f"tw mismatch at {b},{a},{row},{col}: {actual_tw} != {expected_data['tw']}"
                assert np.isclose(actual_th, expected_data["th"], atol=1e-6), f"th mismatch at {b},{a},{row},{col}: {actual_th} != {expected_data['th']}"

    def test_strategy_1d_variation(self, setup_params):
        """
        Strategy 1: Fixed Height, Varying Width
        Boxes have the same height as the anchors, reducing IoU to 1D width matching.
        """
        def anchor_gen(num_anchors):
            alpha = 0.9 / num_anchors
            beta = 0.5
            return [( (k+1)*alpha, beta ) for k in range(num_anchors)]

        def target_gen(anchor_idx, anchors, row, col, h, w):
            aw, ah = anchors[anchor_idx]
            
            # 1. Generate narrow regression targets to prevent edge cases
            tx = np.random.uniform(0.05, 0.95)
            ty = np.random.uniform(0.05, 0.95)
            # Keeping tw, th tight ensures the resulting bw, bh don't drift into another anchor's territory
            tw = np.random.uniform(-0.1, 0.1)
            th = np.random.uniform(-0.1, 0.1)
            
            # 2. Inverse Math to get GT coordinates
            cx = (col + tx) / w
            cy = (row + ty) / h
            bw = aw * np.exp(tw)
            bh = ah * np.exp(th)
            
            # The calculator uses: torch.log(target / (anchor + 1e-7) + 1e-7)
            # We calculate the exact mathematical expected value it will produce
            expected_tw = np.log(bw / (aw + 1e-7) + 1e-7)
            expected_th = np.log(bh / (ah + 1e-7) + 1e-7)
            
            return cx, cy, bw, bh, tx, ty, expected_tw, expected_th

        self._run_exhaustive_test(setup_params, anchor_gen, target_gen)

    def test_strategy_exponential(self, setup_params):
        """
        Strategy 2: Exponential Scaling
        Massive gaps in area between anchors so target box size cannot map to neighbors.
        """
        def anchor_gen(num_anchors):
            # Scale factor of 2. Reverse so max is around 0.8
            base = 0.8 / (2 ** (num_anchors - 1))
            return [( base * (2**k), base * (2**k) ) for k in range(num_anchors)]

        def target_gen(anchor_idx, anchors, row, col, h, w):
            aw, ah = anchors[anchor_idx]
            tx = np.random.uniform(0.05, 0.95)
            ty = np.random.uniform(0.05, 0.95)
            tw = np.random.uniform(-0.1, 0.1)
            th = np.random.uniform(-0.1, 0.1)
            
            cx = (col + tx) / w
            cy = (row + ty) / h
            bw = aw * np.exp(tw)
            bh = ah * np.exp(th)
            
            expected_tw = np.log(bw / (aw + 1e-7) + 1e-7)
            expected_th = np.log(bh / (ah + 1e-7) + 1e-7)
            
            return cx, cy, bw, bh, tx, ty, expected_tw, expected_th

        self._run_exhaustive_test(setup_params, anchor_gen, target_gen)

    def test_strategy_constant_area(self, setup_params):
        """
        Strategy 3: Constant Area, Extreme Aspect Ratios
        All anchors have identical area (0.04) but drastically different aspect ratios.
        """
        def anchor_gen(num_anchors):
            ratios = [1.0, 2.5, 0.4, 4.0, 0.25, 6.0, 0.16]
            return [( 0.2 * r, 0.2 / r ) for r in ratios[:num_anchors]]

        def target_gen(anchor_idx, anchors, row, col, h, w):
            aw, ah = anchors[anchor_idx]
            tx = np.random.uniform(0.05, 0.95)
            ty = np.random.uniform(0.05, 0.95)
            tw = np.random.uniform(-0.1, 0.1)
            th = np.random.uniform(-0.1, 0.1)
            
            cx = (col + tx) / w
            cy = (row + ty) / h
            bw = aw * np.exp(tw)
            bh = ah * np.exp(th)
            
            expected_tw = np.log(bw / (aw + 1e-7) + 1e-7)
            expected_th = np.log(bh / (ah + 1e-7) + 1e-7)
            
            return cx, cy, bw, bh, tx, ty, expected_tw, expected_th

        self._run_exhaustive_test(setup_params, anchor_gen, target_gen)

    def test_strategy_orthogonal_alternation(self, setup_params):
        """
        Strategy 4: Orthogonal Alternation
        Hardcoded, heavily distinct anchor shapes alternating tall and wide.
        """
        def anchor_gen(num_anchors):
            # To absolutely guarantee no collisions, we need extreme differentiation.
            # We'll use 5 highly distinct "profiles":
            # 1: Extreme horizontal sliver
            # 2: Extreme vertical sliver
            # 3: Small square
            # 4: Large horizontal rectangle
            # 5: Large vertical rectangle
            sizes = [
                (0.9, 0.05), # 1. Wide and extremely thin
                (0.05, 0.9), # 2. Tall and extremely thin
                (0.2, 0.2),  # 3. Small square
                (0.8, 0.3),  # 4. Large horizontal
                (0.3, 0.8)   # 5. Large vertical
            ]
            return sizes[:num_anchors]

        def target_gen(anchor_idx, anchors, row, col, h, w):
            aw, ah = anchors[anchor_idx]
            tx = np.random.uniform(0.05, 0.95)
            ty = np.random.uniform(0.05, 0.95)
            tw = np.random.uniform(-0.1, 0.1)
            th = np.random.uniform(-0.1, 0.1)
            
            cx = (col + tx) / w
            cy = (row + ty) / h
            bw = aw * np.exp(tw)
            bh = ah * np.exp(th)
            
            expected_tw = np.log(bw / (aw + 1e-7) + 1e-7)
            expected_th = np.log(bh / (ah + 1e-7) + 1e-7)
            
            return cx, cy, bw, bh, tx, ty, expected_tw, expected_th

        self._run_exhaustive_test(setup_params, anchor_gen, target_gen)

    def test_empty_targets(self, setup_params):
        """
        Verify that an empty target tensor correctly resolves to all zeros.
        """
        batch_size = 4
        num_classes = 5
        h, w = 13, 13
        anchors = [(0.2, 0.2), (0.4, 0.4), (0.6, 0.6)]
        
        calculator = YoloV2TargetCalculator(num_classes, anchors, (h, w))
        
        targets = torch.zeros((0, 6), dtype=torch.float32)
        target_tensor = calculator.compute_targets(targets, batch_size)
        
        assert target_tensor.sum().item() == 0.0
        assert target_tensor.shape == (batch_size, len(anchors) * (5 + num_classes), h, w)