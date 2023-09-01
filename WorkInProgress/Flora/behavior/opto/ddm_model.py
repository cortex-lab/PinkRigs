# %%
import pyddm
import pyddm.plot
import pandas


class DriftAdditiveBias(pyddm.Drift):
    name = "additive drift with bias"
    required_parameters = ["aud_coef", "vis_coef", "stim_bias"]
    required_conditions = ["aud_evidence", "vis_evidence", "stim"]
    def get_drift(self, conditions, **kwargs):
        return self.aud_coef * conditions["aud_evidence"] + self.vis_coef * conditions["vis_evidence"] + (self.stim_bias if conditions["stim"] else 0)


m = pyddm.Model(drift=DriftAdditiveBias(aud_coef=pyddm.Fittable(minval=-1, maxval=1),
                                        vis_coef=pyddm.Fittable(minval=-1, maxval=1),
                                        stim_bias=pyddm.Fittable(minval=-2, maxval=2)), # total bias value
                noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.2, maxval=1)),
                bound=pyddm.BoundConstant(B=1),
                overlay=pyddm.OverlayChain(overlays=[
                    pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=.3)),
                    pyddm.OverlayExponentialMixture(pmixturecoef=pyddm.Fittable(minval=0, maxval=.4),
                                                    rate=1),
                    ]),
                dt=.01, dx=.01, T_dur=4)

pyddm.plot.model_gui(model=m, conditions={"aud_evidence": [-60, 0,60], "vis_evidence": [-40,-20,0,20,40], "stim": [1, 0]})
# %%
