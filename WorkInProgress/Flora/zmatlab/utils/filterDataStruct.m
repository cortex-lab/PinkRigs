function extracted  = filterDataStruct(extracted,toKeep)
% utility function to filter the structure (extracted) that gets spitted out by
% plt.behaviour.getTrainingData 
% toKeep: bool which rows to keep
fn = fieldnames(extracted);
for k=1:numel(fn)
    d = extracted.(fn{k});
    extracted.(fn{k}) = {d{toKeep}}';         
end
end 