"""
Smart Parking AI: Improved Detection Script
ISTE-500/501 Senior Development Project
Team: Dilshod, Walid, Sayed, Meiir

This improved version has two modes:
1. VEHICLE DETECTION MODE - Detects all vehicles (great for demos!)
2. SLOT DEFINITION MODE - Click to define parking spaces

Usage:
    python smart_parking_v2.py image.jpg              # Vehicle detection
    python smart_parking_v2.py image.jpg --define     # Define slots interactively
    python smart_parking_v2.py image.jpg --config slots.json  # Use saved slots
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from pathlib import Path
import argparse


class SmartParkingV2:
    """Improved Smart Parking Detection with multiple modes."""
    
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize with YOLO model."""
        print("🚗 Initializing Smart Parking AI v2...")
        self.model = YOLO(model_path)
        
        # Vehicle classes in COCO: car, motorcycle, bus, truck
        self.vehicle_classes = [2, 3, 5, 7]
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # For interactive slot definition
        self.slots = []
        self.current_points = []
        self.defining_slots = False
        
        print("✅ Model loaded!")
    
    def detect_vehicles(self, image_path, output_path=None, show=True, conf_threshold=0.3):
        """
        MODE 1: Detect all vehicles in the image.
        This is the best mode for demos - clearly shows AI detection working.
        """
        print(f"\n📸 Detecting vehicles in: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = frame.shape[:2]
        
        # Run YOLO detection
        print("  🤖 Running YOLOv8 detection...")
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections
        vehicles = []
        output = frame.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                
                if cls in self.vehicle_classes:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': self.class_names.get(cls, 'vehicle')
                    })
                    
                    # Draw bounding box
                    color = (0, 0, 255)  # Red for vehicles
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with confidence
                    label = f"{self.class_names.get(cls, 'vehicle')} {conf*100:.1f}%"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(output, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(output, label, (x1, y1 - 7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw stats panel
        self._draw_vehicle_stats(output, vehicles)
        
        # Print results
        print(f"\n  ✅ Detection Complete!")
        print(f"     - Vehicles Found: {len(vehicles)}")
        for v in vehicles:
            print(f"       • {v['class']}: {v['confidence']*100:.1f}%")
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, output)
            print(f"     - Saved to: {output_path}")
        
        # Display
        if show:
            # Resize for display if too large
            display = self._resize_for_display(output)
            cv2.imshow('Smart Parking AI - Vehicle Detection', display)
            print("\n  Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return output, vehicles
    
    def detect_with_slots(self, image_path, slots_config, output_path=None, show=True):
        """
        MODE 2: Detect vehicles and check slot occupancy using predefined slots.
        """
        print(f"\n📸 Processing with slot configuration...")
        
        # Load slots from config
        if isinstance(slots_config, str):
            with open(slots_config, 'r') as f:
                slots = json.load(f)
        else:
            slots = slots_config
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run YOLO detection
        results = self.model(frame, conf=0.3, verbose=False)
        
        # Get vehicle bounding boxes
        vehicle_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_boxes.append((x1, y1, x2, y2))
        
        # Check each slot for occupancy
        output = frame.copy()
        occupied_count = 0
        
        for i, slot in enumerate(slots):
            # Get slot polygon points
            points = np.array(slot['points'], dtype=np.int32)
            
            # Check if any vehicle overlaps with this slot
            is_occupied = False
            slot_confidence = 0
            
            for vbox in vehicle_boxes:
                overlap = self._check_overlap(points, vbox)
                if overlap > 0.3:  # 30% overlap threshold
                    is_occupied = True
                    slot_confidence = overlap
                    break
            
            # Draw slot
            if is_occupied:
                color = (0, 0, 255)  # Red
                label = f"#{i+1} OCCUPIED"
                occupied_count += 1
            else:
                color = (0, 255, 0)  # Green
                label = f"#{i+1} AVAILABLE"
            
            # Draw polygon
            cv2.polylines(output, [points], True, color, 3)
            
            # Fill with transparency
            overlay = output.copy()
            cv2.fillPoly(overlay, [points], color)
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
            
            # Draw label
            centroid = points.mean(axis=0).astype(int)
            cv2.putText(output, label, (centroid[0] - 50, centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if is_occupied and slot_confidence > 0:
                conf_text = f"{slot_confidence*100:.1f}%"
                cv2.putText(output, conf_text, (centroid[0] - 20, centroid[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw stats
        self._draw_slot_stats(output, len(slots), occupied_count)
        
        # Save and show
        if output_path:
            cv2.imwrite(output_path, output)
        
        if show:
            display = self._resize_for_display(output)
            cv2.imshow('Smart Parking AI - Slot Detection', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return output
    
    def define_slots_interactive(self, image_path, output_config='slots.json'):
        """
        MODE 3: Interactively define parking slots by clicking.
        
        Instructions:
        - Click 4 corners of each parking slot (clockwise or counter-clockwise)
        - Press 'n' to finish current slot and start next
        - Press 's' to save all slots
        - Press 'u' to undo last point
        - Press 'r' to reset current slot
        - Press 'q' to quit
        """
        print(f"\n🖱️  Interactive Slot Definition Mode")
        print("=" * 50)
        print("Instructions:")
        print("  • Click 4 corners of each parking slot")
        print("  • Press 'n' = finish slot, start next")
        print("  • Press 's' = save all slots to file")
        print("  • Press 'u' = undo last point")
        print("  • Press 'r' = reset current slot")
        print("  • Press 'q' = quit")
        print("=" * 50)
        
        # Read image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.image = self.original_image.copy()
        self.slots = []
        self.current_points = []
        self.defining_slots = True
        
        # Create window and set mouse callback
        cv2.namedWindow('Define Parking Slots')
        cv2.setMouseCallback('Define Parking Slots', self._mouse_callback)
        
        while True:
            # Draw current state
            display = self.image.copy()
            
            # Draw completed slots
            for i, slot in enumerate(self.slots):
                points = np.array(slot['points'], dtype=np.int32)
                cv2.polylines(display, [points], True, (0, 255, 0), 2)
                centroid = points.mean(axis=0).astype(int)
                cv2.putText(display, f"#{i+1}", (centroid[0]-10, centroid[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw current points being defined
            for point in self.current_points:
                cv2.circle(display, point, 5, (0, 0, 255), -1)
            
            if len(self.current_points) > 1:
                for i in range(len(self.current_points) - 1):
                    cv2.line(display, self.current_points[i], self.current_points[i+1], (0, 0, 255), 2)
            
            # Draw instructions
            cv2.putText(display, f"Slots defined: {len(self.slots)} | Current points: {len(self.current_points)}/4",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "n=next slot | s=save | u=undo | r=reset | q=quit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Resize for display
            display = self._resize_for_display(display)
            cv2.imshow('Define Parking Slots', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):  # Next slot
                if len(self.current_points) >= 3:
                    self.slots.append({'points': self.current_points.copy()})
                    print(f"  ✓ Slot #{len(self.slots)} defined")
                    self.current_points = []
                else:
                    print("  ⚠ Need at least 3 points for a slot")
            
            elif key == ord('s'):  # Save
                if len(self.slots) > 0:
                    with open(output_config, 'w') as f:
                        json.dump(self.slots, f, indent=2)
                    print(f"\n  ✅ Saved {len(self.slots)} slots to {output_config}")
                else:
                    print("  ⚠ No slots to save")
            
            elif key == ord('u'):  # Undo
                if len(self.current_points) > 0:
                    self.current_points.pop()
                    print("  ↩ Undid last point")
            
            elif key == ord('r'):  # Reset current
                self.current_points = []
                print("  🔄 Reset current slot")
            
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        
        if len(self.slots) > 0:
            print(f"\n✅ Defined {len(self.slots)} parking slots")
            print(f"   Config saved to: {output_config}")
            print(f"\n   To use: python smart_parking_v2.py {image_path} --config {output_config}")
        
        return self.slots
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for slot definition."""
        if event == cv2.EVENT_LBUTTONDOWN and self.defining_slots:
            # Scale coordinates back if image was resized
            h, w = self.original_image.shape[:2]
            display_h, display_w = self._get_display_size(h, w)
            
            scale_x = w / display_w
            scale_y = h / display_h
            
            actual_x = int(x * scale_x)
            actual_y = int(y * scale_y)
            
            self.current_points.append((actual_x, actual_y))
            print(f"  📍 Point {len(self.current_points)}: ({actual_x}, {actual_y})")
            
            if len(self.current_points) == 4:
                print("  → 4 points reached. Press 'n' to confirm slot or 'r' to reset")
    
    def _check_overlap(self, polygon_points, bbox):
        """Check overlap between a polygon and a bounding box."""
        # Create masks
        x1, y1, x2, y2 = bbox
        
        # Get bounding rect of polygon
        px, py, pw, ph = cv2.boundingRect(polygon_points)
        
        # Quick check - if bounding boxes don't overlap, no need for detailed check
        if x2 < px or x1 > px + pw or y2 < py or y1 > py + ph:
            return 0
        
        # Create a small canvas for overlap calculation
        max_x = max(x2, px + pw) + 10
        max_y = max(y2, py + ph) + 10
        
        # Polygon mask
        poly_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [polygon_points], 255)
        
        # Bbox mask
        bbox_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(poly_mask, bbox_mask)
        
        # Calculate areas
        poly_area = cv2.countNonZero(poly_mask)
        intersection_area = cv2.countNonZero(intersection)
        
        if poly_area == 0:
            return 0
        
        return intersection_area / poly_area
    
    def _draw_vehicle_stats(self, frame, vehicles):
        """Draw statistics panel for vehicle detection mode."""
        height, width = frame.shape[:2]
        
        # Panel
        panel_width = 300
        panel_height = 180
        cv2.rectangle(frame, (width - panel_width - 10, 10), 
                     (width - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - panel_width - 10, 10), 
                     (width - 10, panel_height), (0, 200, 255), 2)
        
        # Title
        cv2.putText(frame, "SMART PARKING AI", 
                   (width - panel_width + 10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # Stats
        stats = [
            f"Vehicles Detected: {len(vehicles)}",
            "",
            "Breakdown:",
        ]
        
        # Count by type
        type_counts = {}
        for v in vehicles:
            vtype = v['class']
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        for vtype, count in type_counts.items():
            stats.append(f"  {vtype}: {count}")
        
        y = 70
        for stat in stats:
            cv2.putText(frame, stat, (width - panel_width + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22
    
    def _draw_slot_stats(self, frame, total_slots, occupied):
        """Draw statistics panel for slot detection mode."""
        height, width = frame.shape[:2]
        available = total_slots - occupied
        
        # Panel
        panel_width = 280
        cv2.rectangle(frame, (width - panel_width - 10, 10), 
                     (width - 10, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - panel_width - 10, 10), 
                     (width - 10, 200), (0, 200, 255), 2)
        
        # Title
        cv2.putText(frame, "SMART PARKING AI", 
                   (width - panel_width + 10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # Stats
        stats = [
            (f"Total Slots: {total_slots}", (255, 255, 255)),
            (f"Occupied: {occupied}", (0, 0, 255)),
            (f"Available: {available}", (0, 255, 0)),
            (f"Occupancy: {occupied/total_slots*100:.1f}%", (255, 255, 255)),
        ]
        
        y = 75
        for text, color in stats:
            cv2.putText(frame, text, (width - panel_width + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 30
    
    def _get_display_size(self, height, width, max_dim=1200):
        """Calculate display size maintaining aspect ratio."""
        if max(height, width) <= max_dim:
            return height, width
        
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        
        return new_height, new_width
    
    def _resize_for_display(self, frame, max_dim=1200):
        """Resize frame for display if too large."""
        height, width = frame.shape[:2]
        new_height, new_width = self._get_display_size(height, width, max_dim)
        
        if (new_height, new_width) != (height, width):
            return cv2.resize(frame, (new_width, new_height))
        return frame


def main():
    """Main function with command line interface."""
    print("=" * 60)
    print("  🅿️  SMART PARKING AI v2")
    print("  Improved Detection with Multiple Modes")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Smart Parking AI with multiple detection modes.")
    parser.add_argument("image", help="Path to the input parking image")
    parser.add_argument("--output", default="detection_result.jpg", help="Output image filename (default: detection_result.jpg)")
    parser.add_argument("--define", action="store_true", help="Interactively define parking spaces")
    parser.add_argument("--config", metavar="FILE", help="Use parking spaces defined in a JSON configuration file")
    args = parser.parse_args()

    image_path = args.image
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    # Initialize detector
    detector = SmartParkingV2()
    
    # Determine mode
    if args.define:
        # Interactive slot definition mode
        detector.define_slots_interactive(image_path)
    
    elif args.config:
        # Use predefined slots
        detector.detect_with_slots(
            image_path,
            args.config,
            output_path=args.output
        )
    
    else:
        # Default: Vehicle detection mode
        detector.detect_vehicles(
            image_path,
            output_path=args.output,
            show=True
        )
    
    print("\n" + "=" * 60)
    if args.define:
        print("  Done!")
    else:
        print(f"  ✅ Done! Check '{args.output}'")
    print("=" * 60)


if __name__ == "__main__":
    main()