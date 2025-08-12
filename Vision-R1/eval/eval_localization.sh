# Evaluation Prompt for Different Models and Tasks 
# Qwen2.5_VL
## COCO
### Locate every item from the category list in the image and output the coordinates in JSON format. The category set includes <category set>.
## ODINW or Visual Grounding
### Locate every <category set> in the image and output the coordinates in JSON format.

# Griffon
## COCO
### Examine the image for any objects from the category set. Report the coordinates of each detected object. The category set includes <category set>.
## ODINW or Visual Grounding
### Locate the exact position of <category set> in the picture, if possible.

# InternVL
## ODINW or Visual Grounding
### Please provide the bounding box coordinate of the region this sentence describes: <ref><category set></ref>
### "Please provide the bounding box coordinates of every <category set> in the image in JSON format as following: [\{\"label\": category_name, \"bbox_2d\": [x1, y1, x2, y2]\}, ...]."
# Ferret
### What are the locations of <category set>?

###################################### INFO ####################################################
# When evaluating the ODINW, the single and pos are required to be added to reproduce the results.
# While for COCO evaluation, they should be removed to be false.
###############################################################################################

torchrun --nproc_per_node 8 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 12355 \
        eval/eval_localization.py \
        --model-path ${1} \
        --model-type ${2} \
        --init tcp://127.0.0.1:12355 \
        --query "Examine the image for any objects from the category set. Report the coordinates of each detected object. The category set includes <category set>." \
        --batch-size 1 \
        --dataset ${3} 
        # --single \
        # --pos \