function status = parseStatusCode(code)
   %%% This function will just parse the code in a more explicit form.
   %%% Should be a cell.
   
   if ischar(code)
       code = cellstr(code);
   end
   
   status = struct();
   codeIndivBits = cell(1,numel(code));
   for cc = 1:numel(code)
       codeIndivBits{cc} = regexp(code{cc},',','split');
       
       switch numel(codeIndivBits{cc})
           case 2
               % spk/ev preprocessing
               status(cc).spk = codeIndivBits{cc}{1};
               status(cc).ev = codeIndivBits{cc}{2};
           case 6
               % alignment
               status(cc).block = codeIndivBits{cc}{1};
               status(cc).frontCam = codeIndivBits{cc}{2};
               status(cc).sideCam = codeIndivBits{cc}{3};
               status(cc).eyeCam = codeIndivBits{cc}{4};
               status(cc).mic = codeIndivBits{cc}{5};
               status(cc).ephys = codeIndivBits{cc}{6};
       end
   end
   
end