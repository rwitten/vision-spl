function [examples_at_step] = load_latents(filename, pos_or_neg) %pos_or_neg = 0 means both types of examples,
                                                                 %pos_or_neg = 1 means only postives
                                                                 %pos_or_neg = -1 means only negatives
   
    num_kernels=5;
    spl= load(filename);
    if abs(pos_or_neg)<1e-4,
        spl = abs(spl);
    elseif pos_or_neg >.5,
        disp 'pos only'
        spl = spl.*(spl>0);
    elseif pos_or_neg<.5, 
        spl = -spl.*(spl<0);
    end

    num_examples = size(spl,2)/num_kernels;
    assert(floor(num_examples)==num_examples);
   
    examples_at_step = ones(size(spl,1),num_kernels); 
    for j=1:num_kernels,
        matrix = spl(:,j:num_kernels:size(spl,2));
        examples_at_step(:,j) = sum(matrix,2);
    end
    
    plot(examples_at_step);
end
