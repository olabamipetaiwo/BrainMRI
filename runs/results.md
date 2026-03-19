Result Interpretation                                                                                                                                                          
                                                                                                                                                                                 
  What the Dice Scores Mean                                                                                                                                                      
                                                                                                                                                                                 
  Dice ranges from 0 (no overlap) to 1 (perfect). In medical segmentation:                                                                                                       
  - > 0.85 — excellent                                                                                                                                                           
  - 0.70–0.85 — good, clinically useful                                                                                                                                          
  - < 0.70 — marginal                  
                                                                                                                                                                                 
  ┌────────────┬─────────────────┬──────────────────────────────────────────────────────────────┐                                                                                
  │   Region   │    Avg Dice     │                           Verdict                            │                                                                                
  ├────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤                                                                                
  │ WT (0.869) │ Whole Tumor     │ Good — model reliably finds where the tumor is               │                                                                                
  ├────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤
  │ TC (0.755) │ Tumor Core      │ Good — reasonable delineation of the core mass               │                                                                                
  ├────────────┼─────────────────┼──────────────────────────────────────────────────────────────┤                                                                                
  │ ET (0.695) │ Enhancing Tumor │ Marginal — struggles with the smallest, most critical region │                                                                                
  └────────────┴─────────────────┴──────────────────────────────────────────────────────────────┘                                                                                
                  
  What the HD95 Scores Mean                                                                                                                                                      
                  
  HD95 measures the 95th percentile of surface distances (in mm) between prediction and ground truth. Lower is better. At 1mm isotropic spacing:                                 
  - < 5mm — good boundary accuracy
  - 5–10mm — moderate, some boundary errors                                                                                                                                      
  - > 10mm — poor boundary delineation     
                                                                                                                                                                                 
  Your averages (ET: 5.32mm, TC: 5.98mm, WT: 5.12mm) are all in the moderate range — the model gets the shape roughly right but has some surface inaccuracies.
                                                                                                                                                                                 
  ---             
  Why ET is the Weakest                                                                                                                                                          
                       
  ET (enhancing tumor) is consistently the hardest region across all BraTS work because:
  - It's the smallest region — very few voxels, so even small prediction errors hurt Dice significantly                                                                          
  - It's heterogeneous in appearance across patients                                                                                                                             
  - The model was only trained for 10 epochs (development default) — ET needs more training iterations to learn fine-grained features                                            
                                                                                                                                                                                 
  Comparison to BraTS State of the Art                                                                                                                                           
                                                                                                                                                                                 
  For context, top BraTS models typically achieve:                                                                                                                               
                                                                                                                                                                                 
  ┌─────────┬────────────┬────────────┐                                                                                                                                          
  │ Region  │  Top SOTA  │ Your Model │
  ├─────────┼────────────┼────────────┤                                                                                                                                          
  │ ET Dice │ ~0.80–0.83 │ 0.695      │
  ├─────────┼────────────┼────────────┤
  │ TC Dice │ ~0.85–0.88 │ 0.755      │                                                                                                                                          
  ├─────────┼────────────┼────────────┤
  │ WT Dice │ ~0.90–0.92 │ 0.869      │                                                                                                                                          
  └─────────┴────────────┴────────────┘                                                                                                                                          
   
  Your model is below SOTA but reasonable given the 10-epoch training. With the full 200–400 epochs recommended in the spec, these numbers would likely improve significantly —  
  especially ET and TC.
                                                                                                                                                                                 
  ---             
  Bottom Line
             
  The model works — it has learned meaningful tumor structure — but it's undertrained. The WT results are already solid. ET is the target for improvement and would benefit most
  from more training epochs and potentially a weighted loss that emphasizes the smaller tumor regions. 