 implement XY alignment and new save format - 
1. If the images are not stacks take the ch1 tiff, align it to the 1080 ch1 tiff in XY and save the aligned version. Save this directly in the flourophore directory - 1 less intermediate subdirectory. so if it was in mCherry its still in mcherry just noe mcherry/subdir or if it was in fig3, its still in fig3 just not fig3/subdir. Name it so its filterName_excNM_valPoc. 
2. If the images are stacks we want to align each to the 1080 and then save only the best match (matched in Z). 
3. If the 1080 acquisition is a stack we want to take slices from this stack every 10 frames. Extract the best match in Z for each from the other stacks as in 2, and then compile these into a stack (10x smaller) and save the fully fully aligned stacks. 
 There should be some code in /from_sam that does this. PLease take a look and extract any useful functions into an existing or new python source file. Lets discuss whether this is sufficient or whether it needs to be improved (possibly we need to do XY alignment before finding best Z match) 

We want to align and save for each of the directories in /data