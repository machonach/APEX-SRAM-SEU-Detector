#!/bin/bash
# Script to merge the single-board and dual-board files for the SEU Detector project
# This script helps you merge the modified single-board files with the original files

echo "==== APEX-SRAM-SEU-Detector File Merger ===="
echo "This script will help you merge the single-board and dual-board files."
echo ""

# Define colors for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if required tools are available
if ! command -v diff &> /dev/null; then
    echo -e "${RED}Error: diff utility is required but not found.${NC}"
    echo "Please install it (e.g., 'sudo apt-get install diffutils' on Debian/Ubuntu)"
    exit 1
fi

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File $1 not found.${NC}"
        echo "Please make sure you're running this script from the APEX-SRAM-SEU-Detector directory."
        exit 1
    fi
}

# Function to compare and merge files interactively
merge_files() {
    local original_file="$1"
    local new_file="$2"
    local output_file="$3"
    local description="$4"
    
    echo -e "\n${BLUE}==== Merging $description ====${NC}"
    echo "Original file: $original_file"
    echo "New file: $new_file"
    echo "Output will be saved as: $output_file"
    
    # Check if files exist
    check_file "$original_file"
    check_file "$new_file"
    
    # Show a diff of the files
    echo -e "\n${YELLOW}Showing differences between files:${NC}"
    diff -u "$original_file" "$new_file"
    
    # Ask user what to do
    echo -e "\n${GREEN}How would you like to merge these files?${NC}"
    echo "1) Keep original file ($original_file)"
    echo "2) Use new file ($new_file)"
    echo "3) Use merge tool (if available)"
    echo "4) Skip this merge"
    read -p "Select option [1-4]: " merge_option
    
    case $merge_option in
        1)
            echo "Keeping original file..."
            cp "$original_file" "$output_file"
            echo -e "${GREEN}Original file kept as $output_file${NC}"
            ;;
        2)
            echo "Using new file..."
            cp "$new_file" "$output_file"
            echo -e "${GREEN}New file copied as $output_file${NC}"
            ;;
        3)
            # Try to find a suitable merge tool
            if command -v meld &> /dev/null; then
                echo "Opening meld for visual merging..."
                meld "$original_file" "$new_file" "$output_file"
            elif command -v kdiff3 &> /dev/null; then
                echo "Opening kdiff3 for visual merging..."
                kdiff3 "$original_file" "$new_file" -o "$output_file"
            elif command -v vimdiff &> /dev/null; then
                echo "Opening vimdiff for merging..."
                cp "$original_file" "$output_file"
                vimdiff "$output_file" "$new_file"
            else
                echo -e "${YELLOW}No suitable merge tool found (meld, kdiff3, vimdiff).${NC}"
                echo "Using new file as a fallback..."
                cp "$new_file" "$output_file"
            fi
            echo -e "${GREEN}Files merged as $output_file${NC}"
            ;;
        4)
            echo "Skipping this merge..."
            return
            ;;
        *)
            echo -e "${RED}Invalid option. Skipping this merge.${NC}"
            return
            ;;
    esac
}

# Function to update references in markdown files
update_references() {
    local file="$1"
    local old_reference="$2"
    local new_reference="$3"
    
    echo -e "\n${BLUE}Updating references in $file${NC}"
    echo "Old reference: $old_reference"
    echo "New reference: $new_reference"
    
    # Check if file exists
    check_file "$file"
    
    # Update references
    sed -i "s/$old_reference/$new_reference/g" "$file"
    echo -e "${GREEN}References updated in $file${NC}"
}

# Main script execution

# 1. Merge setup scripts
merge_files "setup_pi_zero2w.sh" "setup_pi_zero2w_single.sh" "setup_pi_zero2w_merged.sh" "Setup Scripts"

# 2. Merge high-altitude launch guides
merge_files "HIGH_ALTITUDE_LAUNCH.md" "HIGH_ALTITUDE_LAUNCH_SINGLE.md" "HIGH_ALTITUDE_LAUNCH_MERGED.md" "High-Altitude Launch Guides"

# 3. Update references in SINGLE_BOARD_SETUP.md if needed
if [ -f "setup_pi_zero2w_merged.sh" ]; then
    update_references "SINGLE_BOARD_SETUP.md" "setup_pi_zero2w_single.sh" "setup_pi_zero2w_merged.sh"
fi

if [ -f "HIGH_ALTITUDE_LAUNCH_MERGED.md" ]; then
    update_references "SINGLE_BOARD_SETUP.md" "HIGH_ALTITUDE_LAUNCH_SINGLE.md" "HIGH_ALTITUDE_LAUNCH_MERGED.md"
fi

# Final steps
echo -e "\n${GREEN}==== Merge Process Complete ====${NC}"
echo "Next steps:"
echo "1. Review the merged files to ensure they contain all necessary information"
echo "2. Make any additional edits if needed"
echo "3. Rename merged files to replace the originals if desired:"
echo "   - mv setup_pi_zero2w_merged.sh setup_pi_zero2w.sh"
echo "   - mv HIGH_ALTITUDE_LAUNCH_MERGED.md HIGH_ALTITUDE_LAUNCH.md"
echo "4. Delete any unnecessary files once you're satisfied with the merged versions"
echo ""
echo "Happy flying with your Single-Board SEU Detector!"
