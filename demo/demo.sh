IMAGE=$1
COMMAND=$2
echo $IMAGE
echo $COMMAND

echo CUDA_VISIBLE_DEVICES=0 python llava/eval/run_llava.py \
                                --model-path checkpoints/llava-llama-2-13b-chat-lightning-preview-ref-vg-template-finetune-1pe-448-mix4-1ep-OCVR-task-finetune_all_5_3ep \
                                --image-file ${IMAGE:-"demo/1v1.jpg"} \
                                --query ${COMMAND:-"Identify and locate all the objects from the category set in the image. Please provide the coordinates for each detected object. The category set includes dog. The output format for each detected object is class name-[top-left coordinate,bottom-right coordinate] e.g. person-[0.001,0.345,0.111,0.678]. Concatenate them with &."} \
                                --obj "target"

CUDA_VISIBLE_DEVICES=0 python llava/eval/run_llava.py \
                                --model-path checkpoints/llava-llama-2-13b-chat-lightning-preview-ref-vg-template-finetune-1pe-448-mix4-1ep-OCVR-task-finetune_all_5_3ep \
                                --image-file ${IMAGE:-"demo/1v1.jpg"} \
                                --query "${COMMAND:-"Identify and locate all the objects from the category set in the image. Please provide the coordinates for each detected object. The category set includes dog. The output format for each detected object is class name-[top-left coordinate,bottom-right coordinate] e.g. person-[0.001,0.345,0.111,0.678]. Concatenate them with &."}" \
                                --obj "target"