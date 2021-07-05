function dB = rms2dB(rms,rms_Ref,dB_Ref)

dB = dB_Ref + 20*log10(rms/rms_Ref); 

end