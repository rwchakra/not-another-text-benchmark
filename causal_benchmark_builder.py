"""
Video Memory Distortion Benchmark - Dataset Builder
Helps create a minimal benchmark following Strickland & Keil (2011)

Supports all three conditions with complete/incomplete pairs:
1. With Causal Implication
   - Complete: contact shown, trajectory shown
   - Incomplete: contact removed, trajectory shown
2. Without Causal Implication  
   - Complete: contact shown, irrelevant footage (no trajectory)
   - Incomplete: contact removed, irrelevant footage (no trajectory)
3. Scrambled
   - Complete: contact shown, segments in reverse order
   - Incomplete: contact removed, segments in reverse order
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import re


class VideoMemoryBenchmarkBuilder:
    """
    Builds a memory distortion benchmark for video models.
    Implements the experimental design from Strickland & Keil (2011).
    """
    
    def __init__(self, output_dir: str = "benchmark"):
        self.output_dir = Path(output_dir)
        self.videos_dir = self.output_dir / "videos"
        self.annotations_dir = self.output_dir / "annotations"
        
        # Create directories
        for dir_path in [self.videos_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def parse_charades_actions(self, action_string: str) -> List[Dict]:
        """
        Parse Charades action codes into structured format.
        
        Format: "c077 12.10 18.00;c079 11.80 17.30"
        
        Returns list of dicts with action_code, start_time, end_time
        """
        actions = []
        if not action_string:
            return actions
            
        for action in action_string.split(';'):
            parts = action.strip().split()
            if len(parts) >= 3:
                try:
                    actions.append({
                        'action_code': parts[0],
                        'start_time': float(parts[1]),
                        'end_time': float(parts[2])
                    })
                except ValueError:
                    continue
        return actions
    
    def time_to_frame(self, time_seconds: float, fps: float) -> int:
        """Convert time in seconds to frame number"""
        return int(time_seconds * fps)
    
    def frame_to_time(self, frame: int, fps: float) -> float:
        """Convert frame number to time in seconds"""
        return frame / fps
    
    def interactive_frame_viewer(self, video_path: str, start_frame: int = 0) -> Optional[int]:
        """
        Interactive viewer to explore video and mark a frame.
        
        Controls:
            RIGHT arrow / 'd': Next frame
            LEFT arrow / 'a': Previous frame
            'f': Forward 10 frames
            'b': Back 10 frames
            'F': Forward 30 frames
            'B': Back 30 frames
            'g': Go to specific frame (enter in terminal)
            SPACE: Mark current frame and return it
            'q': Quit without marking
        
        Args:
            video_path: Path to source video
            start_frame: Frame to start viewing from
            
        Returns:
            Marked frame number, or None if quit
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        current_frame = min(start_frame, total_frames - 1)
        marked_frame = None
        
        print(f"\n  Controls:")
        print(f"    RIGHT/'d': +1 frame    LEFT/'a': -1 frame")
        print(f"    'f': +10 frames        'b': -10 frames")
        print(f"    'F': +30 frames        'B': -30 frames")
        print(f"    'g': Go to frame       SPACE: Mark frame")
        print(f"    'q': Quit\n")
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame with info
            display = frame.copy()
            time_sec = current_frame / fps
            cv2.putText(display, f"Frame: {current_frame}/{total_frames-1}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Time: {time_sec:.2f}s", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=mark  q=quit  g=goto", 
                       (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
            
            cv2.imshow('Frame Viewer', display)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(' '):  # Space - mark frame
                marked_frame = current_frame
                print(f"    -> Marked frame: {current_frame} ({time_sec:.2f}s)")
                break
            elif key == ord('q'):  # Quit
                break
            elif key == 83 or key == ord('d'):  # Right arrow or 'd'
                current_frame = min(total_frames - 1, current_frame + 1)
            elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                current_frame = max(0, current_frame - 1)
            elif key == ord('f'):  # Forward 10
                current_frame = min(total_frames - 1, current_frame + 10)
            elif key == ord('b'):  # Back 10
                current_frame = max(0, current_frame - 10)
            elif key == ord('F'):  # Forward 30
                current_frame = min(total_frames - 1, current_frame + 30)
            elif key == ord('B'):  # Back 30
                current_frame = max(0, current_frame - 30)
            elif key == ord('g'):  # Go to frame
                cv2.destroyAllWindows()
                try:
                    goto = int(input(f"    Go to frame [0-{total_frames-1}]: "))
                    current_frame = max(0, min(total_frames - 1, goto))
                except ValueError:
                    print("    Invalid frame number")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return marked_frame

    def identify_segments(self, video_path: str) -> Dict:
        """
        Interactive tool to identify key segments in the video.
        
        Uses visual frame viewer to help identify:
        - Video range: Start and end frames to use (for videos with multiple actions)
        - Contact/release moment
        - Trajectory segment (the causal result)
        - Irrelevant footage segment (for "without implication" condition)
        
        All frame numbers are ABSOLUTE (referring to original source video).
        
        Returns:
            Dict with segment information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"\n{'='*60}")
        print(f"SEGMENT IDENTIFICATION")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Total frames: {total_frames} ({total_frames/fps:.2f} seconds)")
        print(f"FPS: {fps:.2f}")
        print(f"\nYou will use the interactive viewer to mark each key frame.")
        print(f"For each step, navigate to the frame and press SPACE to mark it.")
        
        segments = {}
        
        # =====================================================================
        # Step 1: Video Range
        # =====================================================================
        print(f"\n{'='*60}")
        print("STEP 1: VIDEO RANGE")
        print("="*60)
        print("For videos with multiple similar actions (e.g., multiple throws),")
        print("define a range containing only ONE action.")
        
        use_full = input("\nUse full video? (y/n) [n]: ").strip().lower()
        
        if use_full == 'y':
            video_start = 0
            video_end = total_frames - 1
            print(f"Using full video: frames 0-{video_end}")
        else:
            print("\n[1a] Find VIDEO START frame (where single action segment begins)")
            video_start = self.interactive_frame_viewer(video_path, start_frame=0)
            if video_start is None:
                video_start = self._get_frame_input("  Video START frame [0]: ", default=0)
            
            print("\n[1b] Find VIDEO END frame (where single action segment ends)")
            video_end = self.interactive_frame_viewer(video_path, start_frame=video_start)
            if video_end is None:
                video_end = self._get_frame_input(f"  Video END frame [{total_frames-1}]: ", default=total_frames-1)
        
        segments['video_range'] = {'start': video_start, 'end': video_end}
        print(f"\n=> Video range: frames {video_start}-{video_end} "
              f"({(video_end - video_start + 1)/fps:.2f}s)")
        
        # =====================================================================
        # Step 2: Contact Frame
        # =====================================================================
        print(f"\n{'='*60}")
        print("STEP 2: CONTACT FRAME")
        print("="*60)
        print("Find the EXACT frame where contact/release occurs")
        print("(e.g., hand releases ball, foot hits ball)")
        print("This is the critical test frame shown to subjects.")
        
        print("\n[2] Find CONTACT frame")
        contact_frame = self.interactive_frame_viewer(video_path, start_frame=video_start)
        if contact_frame is None:
            contact_frame = self._get_frame_input("  Contact frame: ")
        
        segments['contact_frame'] = contact_frame
        print(f"\n=> Contact frame: {contact_frame} ({contact_frame/fps:.2f}s)")
        
        # =====================================================================
        # Step 3: Contact Removal Range
        # =====================================================================
        print(f"\n{'='*60}")
        print("STEP 3: CONTACT REMOVAL RANGE")
        print("="*60)
        print("Define the range of frames to REMOVE around the contact moment.")
        print("This should include the contact frame and surrounding frames.")
        print("(e.g., frames showing foot approaching ball through foot leaving ball)")
        
        print("\n[3a] Find CONTACT REMOVAL START frame (first frame to remove)")
        suggested_contact_start = max(video_start, contact_frame - 5)
        contact_start = self.interactive_frame_viewer(video_path, start_frame=suggested_contact_start)
        if contact_start is None:
            contact_start = self._get_frame_input(f"  Contact removal START frame [{suggested_contact_start}]: ",
                                                   default=suggested_contact_start)
        
        print("\n[3b] Find CONTACT REMOVAL END frame (last frame to remove)")
        suggested_contact_end = contact_frame + 5
        contact_end = self.interactive_frame_viewer(video_path, start_frame=contact_frame)
        if contact_end is None:
            contact_end = self._get_frame_input(f"  Contact removal END frame [{suggested_contact_end}]: ",
                                                 default=suggested_contact_end)
        
        segments['contact_range'] = {'start': contact_start, 'end': contact_end}
        contact_duration = (contact_end - contact_start + 1) / fps
        frames_removed = contact_end - contact_start + 1
        print(f"\n=> Contact removal range: frames {contact_start}-{contact_end} "
              f"({frames_removed} frames, {contact_duration:.2f}s)")
        
        # =====================================================================
        # Step 4: Irrelevant Footage Range
        # =====================================================================
        print(f"\n{'='*60}")
        print("STEP 4: IRRELEVANT FOOTAGE RANGE")
        print("="*60)
        print("Define segment with UNRELATED footage from same scene")
        print("(e.g., person walking, sitting, reading - NOT the main action)")
        print("This will replace the post-contact footage in 'without implication' condition")
        print("This should NOT overlap with the contact region")
        
        print("\n[4a] Find IRRELEVANT START frame")
        irrel_start = self.interactive_frame_viewer(video_path, start_frame=video_start)
        if irrel_start is None:
            irrel_start = self._get_frame_input("  Irrelevant START frame: ")
        
        print("\n[4b] Find IRRELEVANT END frame")
        irrel_end = self.interactive_frame_viewer(video_path, start_frame=irrel_start)
        if irrel_end is None:
            irrel_end = self._get_frame_input("  Irrelevant END frame: ")
        
        segments['irrelevant'] = {'start': irrel_start, 'end': irrel_end}
        irrel_duration = (irrel_end - irrel_start + 1) / fps
        print(f"\n=> Irrelevant: frames {irrel_start}-{irrel_end} ({irrel_duration:.2f}s)")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("="*60)
        print(f"Video range:     frames {segments['video_range']['start']}-{segments['video_range']['end']} "
              f"({(segments['video_range']['end'] - segments['video_range']['start'] + 1)/fps:.2f}s)")
        print(f"Contact frame:   frame {segments['contact_frame']} ({segments['contact_frame']/fps:.2f}s)")
        print(f"Contact removal: frames {segments['contact_range']['start']}-{segments['contact_range']['end']} "
              f"({(segments['contact_range']['end'] - segments['contact_range']['start'] + 1)} frames)")
        print(f"Irrelevant:      frames {segments['irrelevant']['start']}-{segments['irrelevant']['end']} "
              f"({(segments['irrelevant']['end'] - segments['irrelevant']['start'] + 1)/fps:.2f}s)")
        
        confirm = input("\nProceed with these segments? (y/n) [y]: ").strip().lower()
        if confirm == 'n':
            print("Cancelled.")
            return None
        
        return segments
    
    def _get_frame_input(self, prompt: str, default: int = None) -> int:
        """Helper to get frame number input from user"""
        while True:
            try:
                user_input = input(prompt).strip()
                if user_input == "" and default is not None:
                    return default
                return int(user_input)
            except ValueError:
                print("Please enter a valid frame number")
    
    def _get_video_properties(self, video_path: str) -> Tuple[float, int, int, int]:
        """Get video properties: fps, width, height, total_frames"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, width, height, total_frames
    
    def _create_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """Create a video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # =========================================================================
    # WITH IMPLICATION CONDITION
    # =========================================================================
    
    def create_complete_video_with_implication(self,
                                                video_path: str,
                                                segments: Dict,
                                                output_path: str) -> Dict:
        """
        Complete video for WITH IMPLICATION condition.
        
        Structure:
        [Pre-contact] -> [CONTACT SHOWN] -> [Post-contact to end]
        
        Contact IS shown. Full video through to end.
        """
        fps, width, height, total_frames = self._get_video_properties(video_path)
        
        video_start = segments['video_range']['start']
        video_end = segments['video_range']['end']
        
        cap = cv2.VideoCapture(video_path)
        out = self._create_video_writer(output_path, fps, width, height)
        
        print(f"\nCreating COMPLETE video WITH IMPLICATION...")
        print(f"Range: frames {video_start}-{video_end}")
        print(f"Contact: SHOWN")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start)
        current_frame = video_start
        frames_written = 0
        
        while cap.isOpened() and current_frame <= video_end:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
            current_frame += 1
        
        cap.release()
        out.release()
        
        print(f"[OK] Created: {output_path}")
        print(f"  Frames written: {frames_written}")
        
        return {'frames_written': frames_written}

    def create_incomplete_video_with_implication(self, 
                                                  video_path: str,
                                                  segments: Dict,
                                                  output_path: str) -> Dict:
        """
        Incomplete video for WITH IMPLICATION condition.
        
        Structure:
        [Pre-contact] -> [GAP - contact removed] -> [Post-contact to end]
        
        Contact is REMOVED. Post-contact footage IS shown.
        This should induce false memories of seeing contact.
        """
        fps, width, height, total_frames = self._get_video_properties(video_path)
        
        video_start = segments['video_range']['start']
        video_end = segments['video_range']['end']
        contact_start = segments['contact_range']['start']
        contact_end = segments['contact_range']['end']
        
        cap = cv2.VideoCapture(video_path)
        out = self._create_video_writer(output_path, fps, width, height)
        
        print(f"\nCreating INCOMPLETE video WITH IMPLICATION...")
        print(f"Range: frames {video_start}-{video_end}")
        print(f"Removing frames {contact_start}-{contact_end} (contact region)")
        print(f"Contact: REMOVED")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start)
        current_frame = video_start
        frames_written = 0
        
        while cap.isOpened() and current_frame <= video_end:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames in the contact region
            if not (contact_start <= current_frame <= contact_end):
                out.write(frame)
                frames_written += 1
            
            current_frame += 1
        
        cap.release()
        out.release()
        
        frames_removed = contact_end - contact_start + 1
        print(f"[OK] Created: {output_path}")
        print(f"  Frames written: {frames_written}")
        print(f"  Frames removed: {frames_removed}")
        
        return {
            'frames_written': frames_written,
            'frames_removed': frames_removed,
            'remove_range': (contact_start, contact_end)
        }

    # =========================================================================
    # WITHOUT IMPLICATION CONDITION
    # =========================================================================
    
    def create_complete_video_without_implication(self,
                                                   video_path: str,
                                                   segments: Dict,
                                                   output_path: str) -> Dict:
        """
        Complete video for WITHOUT IMPLICATION condition.
        
        Structure:
        [Pre-contact] -> [CONTACT SHOWN] -> [Irrelevant footage]
        
        Contact IS shown. Post-contact replaced with irrelevant footage.
        """
        fps, width, height, total_frames = self._get_video_properties(video_path)
        
        video_start = segments['video_range']['start']
        contact_end = segments['contact_range']['end']
        contact_start = segments['contact_range']['start']
        irrel_start = segments['irrelevant']['start']
        irrel_end = segments['irrelevant']['end']
        
        cap = cv2.VideoCapture(video_path)
        
        # Read irrelevant frames into memory
        irrelevant_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, irrel_start)
        for i in range(irrel_end - irrel_start + 1):
            ret, frame = cap.read()
            if ret:
                irrelevant_frames.append(frame)
        
        out = self._create_video_writer(output_path, fps, width, height)
        
        print(f"\nCreating COMPLETE video WITHOUT IMPLICATION...")
        print(f"Pre-contact + contact: frames {video_start}-{contact_end}")
        print(f"Then irrelevant footage: frames {irrel_start}-{irrel_end}")
        print(f"Contact: SHOWN")
        print(f"Post-contact: REPLACED with irrelevant footage")
        
        # Write pre-contact and contact frames (up to contact_end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start)
        current_frame = video_start
        frames_written = 0
        
        while cap.isOpened() and current_frame <= contact_end:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
            current_frame += 1
        
        # Write irrelevant frames instead of post-contact
        irrel_frames_written = 0
        for frame in irrelevant_frames:
            out.write(frame)
            frames_written += 1
            irrel_frames_written += 1
        
        cap.release()
        out.release()
        
        print(f"[OK] Created: {output_path}")
        print(f"  Frames written: {frames_written}")
        print(f"  Irrelevant frames used: {irrel_frames_written}")
        
        return {
            'frames_written': frames_written,
            'irrelevant_frames_used': irrel_frames_written
        }

    def create_incomplete_video_without_implication(self,
                                                     video_path: str,
                                                     segments: Dict,
                                                     output_path: str) -> Dict:
        """
        Incomplete video for WITHOUT IMPLICATION condition.
        
        Structure:
        [Pre-contact] -> [GAP - contact removed] -> [Irrelevant footage]
        
        Contact is REMOVED. Post-contact replaced with irrelevant footage.
        This should NOT induce false memories (no causal implication).
        """
        fps, width, height, total_frames = self._get_video_properties(video_path)
        
        video_start = segments['video_range']['start']
        contact_start = segments['contact_range']['start']
        contact_end = segments['contact_range']['end']
        irrel_start = segments['irrelevant']['start']
        irrel_end = segments['irrelevant']['end']
        
        cap = cv2.VideoCapture(video_path)
        
        # Read irrelevant frames into memory
        irrelevant_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, irrel_start)
        for i in range(irrel_end - irrel_start + 1):
            ret, frame = cap.read()
            if ret:
                irrelevant_frames.append(frame)
        
        out = self._create_video_writer(output_path, fps, width, height)
        
        print(f"\nCreating INCOMPLETE video WITHOUT IMPLICATION...")
        print(f"Pre-contact: frames {video_start}-{contact_start - 1}")
        print(f"Removing frames {contact_start}-{contact_end} (contact region)")
        print(f"Then irrelevant footage: frames {irrel_start}-{irrel_end}")
        print(f"Contact: REMOVED")
        print(f"Post-contact: REPLACED with irrelevant footage")
        
        # Write pre-contact frames (up to contact_start)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start)
        current_frame = video_start
        frames_written = 0
        
        while cap.isOpened() and current_frame < contact_start:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
            current_frame += 1
        
        # Write irrelevant frames instead of contact + post-contact
        irrel_frames_written = 0
        for frame in irrelevant_frames:
            out.write(frame)
            frames_written += 1
            irrel_frames_written += 1
        
        cap.release()
        out.release()
        
        frames_removed = contact_end - contact_start + 1
        print(f"[OK] Created: {output_path}")
        print(f"  Frames written: {frames_written}")
        print(f"  Frames removed: {frames_removed}")
        print(f"  Irrelevant frames used: {irrel_frames_written}")
        
        return {
            'frames_written': frames_written,
            'frames_removed': frames_removed,
            'irrelevant_frames_used': irrel_frames_written
        }

    # =========================================================================
    # SCRAMBLED CONDITION
    # =========================================================================
    
    def create_complete_video_scrambled(self,
                                         video_path: str,
                                         segments: Dict,
                                         output_path: str,
                                         n_segments: int = 4) -> Dict:
        """
        Complete video for SCRAMBLED condition.
        
        Contact IS shown. Segments in reverse order to disrupt causal coherence.
        """
        fps, width, height, total_frames = self._get_video_properties(video_path)
        
        video_start = segments['video_range']['start']
        video_end = segments['video_range']['end']
        
        cap = cv2.VideoCapture(video_path)
        
        # Read all frames in range
        all_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start)
        current_frame = video_start
        
        while cap.isOpened() and current_frame <= video_end:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
            current_frame += 1
        
        cap.release()
        
        # Divide into n_segments
        n_frames = len(all_frames)
        segment_size = n_frames // n_segments
        
        video_segments = []
        for i in range(n_segments):
            start_idx = i * segment_size
            if i == n_segments - 1:
                end_idx = n_frames
            else:
                end_idx = (i + 1) * segment_size
            video_segments.append(all_frames[start_idx:end_idx])
        
        print(f"\nCreating COMPLETE video SCRAMBLED...")
        print(f"Range: frames {video_start}-{video_end}")
        print(f"Total frames: {n_frames}")
        print(f"Divided into {n_segments} segments:")
        for i, seg in enumerate(video_segments):
            print(f"  Segment {i+1}: {len(seg)} frames")
        print(f"Playing in REVERSE order: {n_segments} -> 1")
        print(f"Contact: SHOWN (within scrambled segments)")
        
        # Write in reverse order
        out = self._create_video_writer(output_path, fps, width, height)
        
        frames_written = 0
        for segment in reversed(video_segments):
            for frame in segment:
                out.write(frame)
                frames_written += 1
        
        out.release()
        
        print(f"[OK] Created: {output_path}")
        print(f"  Total frames: {frames_written}")
        
        return {
            'n_segments': n_segments,
            'frames_written': frames_written
        }

    def create_incomplete_video_scrambled(self,
                                           video_path: str,
                                           segments: Dict,
                                           output_path: str,
                                           n_segments: int = 4) -> Dict:
        """
        Incomplete video for SCRAMBLED condition.
        
        Contact is REMOVED. Segments in reverse order to disrupt causal coherence.
        """
        fps, width, height, total_frames = self._get_video_properties(video_path)
        
        video_start = segments['video_range']['start']
        video_end = segments['video_range']['end']
        contact_start = segments['contact_range']['start']
        contact_end = segments['contact_range']['end']
        
        cap = cv2.VideoCapture(video_path)
        
        # Read all frames except contact region
        all_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start)
        current_frame = video_start
        
        while cap.isOpened() and current_frame <= video_end:
            ret, frame = cap.read()
            if not ret:
                break
            
            if not (contact_start <= current_frame <= contact_end):
                all_frames.append(frame)
            
            current_frame += 1
        
        cap.release()
        
        # Divide into n_segments
        n_frames = len(all_frames)
        segment_size = n_frames // n_segments
        
        video_segments = []
        for i in range(n_segments):
            start_idx = i * segment_size
            if i == n_segments - 1:
                end_idx = n_frames
            else:
                end_idx = (i + 1) * segment_size
            video_segments.append(all_frames[start_idx:end_idx])
        
        frames_removed = contact_end - contact_start + 1
        print(f"\nCreating INCOMPLETE video SCRAMBLED...")
        print(f"Range: frames {video_start}-{video_end}")
        print(f"Removing frames {contact_start}-{contact_end} (contact region)")
        print(f"Total frames (excluding contact): {n_frames}")
        print(f"Divided into {n_segments} segments:")
        for i, seg in enumerate(video_segments):
            print(f"  Segment {i+1}: {len(seg)} frames")
        print(f"Playing in REVERSE order: {n_segments} -> 1")
        print(f"Contact: REMOVED")
        
        # Write in reverse order
        out = self._create_video_writer(output_path, fps, width, height)
        
        frames_written = 0
        for segment in reversed(video_segments):
            for frame in segment:
                out.write(frame)
                frames_written += 1
        
        out.release()
        
        print(f"[OK] Created: {output_path}")
        print(f"  Total frames: {frames_written}")
        print(f"  Frames removed: {frames_removed}")
        
        return {
            'n_segments': n_segments,
            'frames_written': frames_written,
            'frames_removed': frames_removed
        }

    # =========================================================================
    # MAIN BUILD METHOD
    # =========================================================================
    
    def build_test_item(self,
                       source_video_path: str,
                       video_id: str,
                       action_label: str,
                       segments: Dict = None) -> Dict:
        """
        Complete pipeline to build one test item with ALL conditions.
        
        Creates 6 videos:
        - with_implication_complete.mp4
        - with_implication_incomplete.mp4
        - without_implication_complete.mp4
        - without_implication_incomplete.mp4
        - scrambled_complete.mp4
        - scrambled_incomplete.mp4
        
        Args:
            source_video_path: Path to source video
            video_id: Unique ID for this video
            action_label: Action type description
            segments: Pre-defined segments dict, or None for interactive
            
        Returns:
            Test item metadata dict
        """
        print(f"\n{'='*60}")
        print(f"Building test item: {video_id}")
        print(f"Action: {action_label}")
        print(f"Source: {source_video_path}")
        print(f"{'='*60}")
        
        # Validate source exists
        if not Path(source_video_path).exists():
            print(f"[ERROR] Source video not found: {source_video_path}")
            return None
        
        # Step 1: Identify segments if not provided
        if segments is None:
            segments = self.identify_segments(source_video_path)
            if segments is None:
                print("[ERROR] Segment identification cancelled. Skipping.")
                return None
        
        contact_frame = segments['contact_frame']
        contact_range = segments['contact_range']
        
        # Step 2: Get video metadata
        fps, width, height, total_frames = self._get_video_properties(source_video_path)
        duration = total_frames / fps
        
        # Get video range for output duration calculation
        video_range = segments.get('video_range', {'start': 0, 'end': total_frames - 1})
        
        # =====================================================================
        # WITH IMPLICATION CONDITION
        # =====================================================================
        complete_with_impl_path = self.videos_dir / f"{video_id}_with_implication_complete.mp4"
        incomplete_with_impl_path = self.videos_dir / f"{video_id}_with_implication_incomplete.mp4"
        
        with_impl_complete_info = self.create_complete_video_with_implication(
            source_video_path,
            segments,
            str(complete_with_impl_path)
        )
        
        with_impl_incomplete_info = self.create_incomplete_video_with_implication(
            source_video_path,
            segments,
            str(incomplete_with_impl_path)
        )
        
        # =====================================================================
        # WITHOUT IMPLICATION CONDITION
        # =====================================================================
        complete_without_impl_path = self.videos_dir / f"{video_id}_without_implication_complete.mp4"
        incomplete_without_impl_path = self.videos_dir / f"{video_id}_without_implication_incomplete.mp4"
        
        without_impl_complete_info = self.create_complete_video_without_implication(
            source_video_path,
            segments,
            str(complete_without_impl_path)
        )
        
        without_impl_incomplete_info = self.create_incomplete_video_without_implication(
            source_video_path,
            segments,
            str(incomplete_without_impl_path)
        )
        
        # =====================================================================
        # SCRAMBLED CONDITION
        # =====================================================================
        complete_scrambled_path = self.videos_dir / f"{video_id}_scrambled_complete.mp4"
        incomplete_scrambled_path = self.videos_dir / f"{video_id}_scrambled_incomplete.mp4"
        
        scrambled_complete_info = self.create_complete_video_scrambled(
            source_video_path,
            segments,
            str(complete_scrambled_path)
        )
        
        scrambled_incomplete_info = self.create_incomplete_video_scrambled(
            source_video_path,
            segments,
            str(incomplete_scrambled_path)
        )
        
        # =====================================================================
        # Build test item metadata
        # =====================================================================
        test_item = {
            "video_id": video_id,
            "action": action_label,
            "source_video": source_video_path,
            "metadata": {
                "source_duration_seconds": duration,
                "fps": fps,
                "source_total_frames": total_frames,
                "resolution": [width, height]
            },
            "segments": {
                "video_range": segments.get('video_range', {'start': 0, 'end': total_frames - 1}),
                "contact_frame": contact_frame,
                "contact_time": contact_frame / fps,
                "contact_range": contact_range,
                "irrelevant": segments['irrelevant']
            },
            "videos": {
                "with_implication": {
                    "complete": {
                        "path": str(complete_with_impl_path.relative_to(self.output_dir)),
                        "contact_shown": True,
                        "trajectory_shown": True,
                        "frames_written": with_impl_complete_info['frames_written']
                    },
                    "incomplete": {
                        "path": str(incomplete_with_impl_path.relative_to(self.output_dir)),
                        "contact_shown": False,
                        "trajectory_shown": True,
                        "frames_written": with_impl_incomplete_info['frames_written'],
                        "frames_removed": with_impl_incomplete_info['frames_removed']
                    }
                },
                "without_implication": {
                    "complete": {
                        "path": str(complete_without_impl_path.relative_to(self.output_dir)),
                        "contact_shown": True,
                        "trajectory_shown": False,
                        "frames_written": without_impl_complete_info['frames_written']
                    },
                    "incomplete": {
                        "path": str(incomplete_without_impl_path.relative_to(self.output_dir)),
                        "contact_shown": False,
                        "trajectory_shown": False,
                        "frames_written": without_impl_incomplete_info['frames_written'],
                        "frames_removed": without_impl_incomplete_info['frames_removed']
                    }
                },
                "scrambled": {
                    "complete": {
                        "path": str(complete_scrambled_path.relative_to(self.output_dir)),
                        "contact_shown": True,
                        "segments_reversed": True,
                        "n_segments": scrambled_complete_info['n_segments'],
                        "frames_written": scrambled_complete_info['frames_written']
                    },
                    "incomplete": {
                        "path": str(incomplete_scrambled_path.relative_to(self.output_dir)),
                        "contact_shown": False,
                        "segments_reversed": True,
                        "n_segments": scrambled_incomplete_info['n_segments'],
                        "frames_written": scrambled_incomplete_info['frames_written'],
                        "frames_removed": scrambled_incomplete_info['frames_removed']
                    }
                }
            },
            "expected_results": {
                "hypothesis": "False alarm rate on contact frame should be highest for 'with_implication' condition",
                "predictions": {
                    "with_implication_complete": "High correct 'yes' (actually saw contact)",
                    "with_implication_incomplete": "High false alarm 'yes' (didn't see but implied)",
                    "without_implication_complete": "High correct 'yes' (actually saw contact)",
                    "without_implication_incomplete": "Low false alarm (no causal implication)",
                    "scrambled_complete": "High correct 'yes' (actually saw contact)",
                    "scrambled_incomplete": "Low false alarm (causal coherence disrupted)"
                }
            }
        }
        
        # Summary
        print(f"\n{'='*60}")
        print(f"[OK] Test item completed: {video_id}")
        print(f"{'='*60}")
        print(f"Videos created:")
        print(f"  WITH IMPLICATION:")
        print(f"    - {complete_with_impl_path.name}")
        print(f"    - {incomplete_with_impl_path.name}")
        print(f"  WITHOUT IMPLICATION:")
        print(f"    - {complete_without_impl_path.name}")
        print(f"    - {incomplete_without_impl_path.name}")
        print(f"  SCRAMBLED:")
        print(f"    - {complete_scrambled_path.name}")
        print(f"    - {incomplete_scrambled_path.name}")
        
        return test_item
    
    def save_benchmark(self, test_items: List[Dict], filename: str = "benchmark.json"):
        """Save benchmark metadata to JSON"""
        output_path = self.annotations_dir / filename
        
        # Filter out None items
        test_items = [item for item in test_items if item is not None]
        
        benchmark_data = {
            "version": "2.0",
            "name": "Video Memory Distortion Benchmark",
            "description": "Tests whether video models exhibit human-like memory distortions from causal inferences",
            "reference": {
                "paper": "Strickland & Keil (2011)",
                "title": "Event Completion: Event Based Inferences Distort Memory in a Matter of Seconds",
                "journal": "Cognition",
                "volume": 121,
                "issue": 3,
                "pages": "409-415",
                "doi": "10.1016/j.cognition.2011.04.007"
            },
            "methodology": {
                "scrambled_segments": 4,
                "contact_removal": "Manually annotated per video"
            },
            "conditions": {
                "with_implication": {
                    "description": "Tests event completion - causal trajectory implies contact occurred",
                    "complete": "Contact shown, trajectory shown",
                    "incomplete": "Contact removed, trajectory shown - should induce false memories"
                },
                "without_implication": {
                    "description": "Control - no causal trajectory to imply contact",
                    "complete": "Contact shown, trajectory replaced with irrelevant footage",
                    "incomplete": "Contact removed, trajectory replaced - should NOT induce false memories"
                },
                "scrambled": {
                    "description": "Control - temporal order disrupted to break causal coherence",
                    "complete": "Contact shown, segments in reverse order",
                    "incomplete": "Contact removed, segments in reverse order"
                }
            },
            "n_videos": len(test_items),
            "test_items": test_items
        }
        
        with open(output_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"[OK] Benchmark saved: {output_path}")
        print(f"  Total test items: {len(test_items)}")
        print(f"{'='*60}")
        
        return output_path


def build_from_charades(builder: VideoMemoryBenchmarkBuilder,
                        charades_video_path: str,
                        video_id: str,
                        action_label: str,
                        video_range: Tuple[int, int],
                        contact_frame: int,
                        contact_range: Tuple[int, int],
                        irrelevant_range: Tuple[int, int]) -> Dict:
    """
    Helper function to build test item from Charades video with pre-specified segments.
    
    All frame numbers are ABSOLUTE (referring to original source video frames).
    
    Args:
        builder: VideoMemoryBenchmarkBuilder instance
        charades_video_path: Path to the Charades video file
        video_id: Charades video ID (e.g., "75M1Z")
        action_label: Description of the action (e.g., "throwing ball up")
        video_range: (start_frame, end_frame) - range to use from source video
        contact_frame: Frame number of the exact contact/release moment (test frame)
        contact_range: (start_frame, end_frame) - frames to remove (contact region)
        irrelevant_range: (start_frame, end_frame) - replacement footage
    """
    segments = {
        'video_range': {'start': video_range[0], 'end': video_range[1]},
        'contact_frame': contact_frame,
        'contact_range': {'start': contact_range[0], 'end': contact_range[1]},
        'irrelevant': {'start': irrelevant_range[0], 'end': irrelevant_range[1]}
    }
    
    return builder.build_test_item(
        source_video_path=charades_video_path,
        video_id=video_id,
        action_label=action_label,
        segments=segments
    )


# Example usage
if __name__ == "__main__":
    print("""
    === Video Memory Distortion Benchmark Builder ===
    
    This tool creates stimuli following Strickland & Keil (2011).
    
    For each video, it creates 6 output videos:
    
    WITH IMPLICATION CONDITION:
      - {id}_with_implication_complete.mp4    (contact + trajectory)
      - {id}_with_implication_incomplete.mp4  (no contact + trajectory)
    
    WITHOUT IMPLICATION CONDITION:
      - {id}_without_implication_complete.mp4    (contact + irrelevant)
      - {id}_without_implication_incomplete.mp4  (no contact + irrelevant)
    
    SCRAMBLED CONDITION:
      - {id}_scrambled_complete.mp4    (contact + reversed segments)
      - {id}_scrambled_incomplete.mp4  (no contact + reversed segments)
    
    Usage:
    
    1. Interactive mode:
       builder = VideoMemoryBenchmarkBuilder(output_dir="benchmark")
       item = builder.build_test_item(
           source_video_path="path/to/video.mp4",
           video_id="my_video",
           action_label="throwing ball"
       )
       builder.save_benchmark([item])
    
    2. With pre-specified segments:
       builder = VideoMemoryBenchmarkBuilder(output_dir="benchmark")
       item = build_from_charades(
           builder=builder,
           charades_video_path="path/to/75M1Z.mp4",
           video_id="75M1Z",
           action_label="throwing ball up",
           video_range=(450, 620),
           contact_frame=554,             # Exact moment of contact (test frame)
           contact_range=(550, 565),      # Frames to remove (contact region)
           irrelevant_range=(700, 740)    # Replacement footage for 'without implication'
       )
       builder.save_benchmark([item])
    """)
