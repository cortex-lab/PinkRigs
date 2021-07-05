function rms = dB2rms(dB,dB_Ref,rms_Ref)

rms = rms_Ref*10.^((dB-dB_Ref)/20);

end