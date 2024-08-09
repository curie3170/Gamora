# python elsage_gnn_multitask_inference_xout123.py --root /home/curie/masGen/DataGen/dataset16 --wandb #this is (x, [out1, out2, out3])
# python elsage_gnn_multitask_inference_x.py --root /home/curie/masGen/DataGen/dataset16 --wandb
# python elsage_gnn_multitask_inference_out123.py --root /home/curie/masGen/DataGen/dataset16 --wandb
# python elsage_gnn_multitask_inference_xout123_cone.py --root /home/curie/masGen/DataGen/dataset16 --PO_bit 0 --wandb #predicting only 1 bit(PO_bit) from corresponding cone
# python elsage_gnn_multitask_inference_xout123_padding.py --root /home/curie/masGen/DataGen/dataset16 --wandb

# python hoga_main_gamora.py --num_hops 4 --save_model
# python hoga_main_gamora.py --num_hops 6 --save_model
# python hoga_main_gamora.py --num_hops 7 --save_model

# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 1 --model_path models/hoga_mult8_mult_1.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 2 --model_path models/hoga_mult8_mult_2.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 3 --model_path models/hoga_mult8_mult_3.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 4 --model_path models/hoga_mult8_mult_4.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 5 --model_path models/hoga_mult8_mult_5.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 6 --model_path models/hoga_mult8_mult_6.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 7 --model_path models/hoga_mult8_mult_7.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 8 --model_path models/hoga_mult8_mult_8.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --wandb


# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 1 --model_path models/hoga_mult8_mult_1.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 2 --model_path models/hoga_mult8_mult_2.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 3 --model_path models/hoga_mult8_mult_3.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 4 --model_path models/hoga_mult8_mult_4.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 5 --model_path models/hoga_mult8_mult_5.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 6 --model_path models/hoga_mult8_mult_6.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 7 --model_path models/hoga_mult8_mult_7.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 8 --model_path models/hoga_mult8_mult_8.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb


# python elsage_hoga_multitask_inference_xout123.py --num_hops 3 --model_path models/hoga_mult8_mult_3.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 3 --model_path models/hoga_mult8_mult_3.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_gnn_multitask_inference_xout123.py --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_gnn_multitask_inference_xout123_padding.py --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb

# [0] python elsage_hoga_multitask_inference_xout123.py --num_hops 3 --model_path models/hoga_mult8_mult_3.pt --root /home/curie/masGen/DataGen/dataset16  --highest_order 16 --batch_size 32 --datatype logic --wandb
# [1] python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 3 --model_path models/hoga_mult8_mult_3.pt --root /home/curie/masGen/DataGen/dataset16  --highest_order 16 --batch_size 32 --datatype logic --wandb
# [2] python elsage_gnn_multitask_inference_xout123.py --root /home/curie/masGen/DataGen/dataset16  --highest_order 16 --batch_size 32 --datatype logic --wandb
# [5] python elsage_gnn_multitask_inference_xout123_padding.py --root /home/curie/masGen/DataGen/dataset16  --highest_order 16 --batch_size 32 --datatype logic --wandb

#python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 1 --model_path models/hoga_mult8_mult_1.pt --root /home/curie/masGen/DataGen/dataset16  --highest_order 16 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123.py --num_hops 4 --model_path models/hoga_mult8_mult_4.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype logic --wandb
# python elsage_hoga_multitask_inference_xout123.py --num_hops 4 --model_path models/hoga_mult8_mult_4.pt --root /home/curie/masGen/DataGen/dataset8  --highest_order 8 --batch_size 32 --datatype xor --wandb

# python elsage_gnn_multitask_inference_xout123_padding.py --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --batch_size 32 --datatype xor --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 1 --model_path models/hoga_mult8_mult_1.pt --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --batch_size 32 --datatype xor --wandb

# python elsage_hoga_multitask_inference_xout123.py --root /home/curie/masGen/DataGen/dataset8 --highest_order 8 --batch_size 32 --datatype xor --wandb
# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 1 --model_path models/hoga_mult8_mult_1.pt --root /home/curie/masGen/DataGen/dataset8 --highest_order 8 --batch_size 32 --datatype xor --wandb

# python elsage_gnn_multitask_inference_xout123_padding.py --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --batch_size 32 --datatype xor --wandb
# python elsage_gnn_multitask_inference_xout123.py --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --batch_size 32 --datatype xor --wandb

# python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 1 --model_path models/hoga_mult8_mult_1.pt --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --batch_size 32 --datatype xor --wandb
# python elsage_hoga_multitask_inference_xout123.py --num_hops 1 --model_path models/hoga_mult8_mult_1.pt --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --batch_size 32 --datatype xor --wandb

# python hoga_main_gamora.py --num_hops 1 --save_model --bits 16
#python elsage_hoga_multitask_inference_xout123_padding.py --num_hops 1 --model_path models/hoga_mult32_mult_1.pt --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --batch_size 32 --datatype xor --wandb

#python elsage_gnn_multitask_inference_xout123.py --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --datatype xor --wandb
python elsage_gnn_multitask_inference_xout123_xoraccum.py --root /home/curie/masGen/DataGen/dataset16 --highest_order 16 --datatype xor_accum --wandb
