from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image


def test_m8_runs_both_branches_per_face_serially_and_isolates_errors():
    from web import frontalize_experiment as experiment

    crops=[Image.new("RGB",(24,32),(value,20,30)) for value in (40,80)]
    views=SimpleNamespace(faces=[SimpleNamespace(face_id=i+1,detection=SimpleNamespace(cropped_face=crop))
                                 for i,crop in enumerate(crops)])
    events=[]; calls=0
    class Frontalizer:
        def correct_roll(self,image,threshold_deg):
            events.append(("roll_estimate",image.getpixel((0,0))[0]))
            return SimpleNamespace(image=image.copy(),yaw_deg=20.,pitch_deg=0.,roll_deg=25.,
                                   applied=True,rotation_deg=-25.,threshold_deg=threshold_deg,time_ms=2.)
    def analyze(data,**kwargs):
        nonlocal calls
        calls+=1; image=Image.open(BytesIO(data)).convert("RGB"); marker=image.getpixel((0,0))[0]
        events.append(("analyze",marker))
        if calls==2: raise RuntimeError("branch failure")
        return {"faces":[]}
    with patch("services.frontalization.get_tddfa_v2_frontalizer",return_value=Frontalizer()), \
         patch("web.orchestrator.analyze",side_effect=analyze):
        state=experiment.run_frontalization_experiment(views,emoca={"mode":"real"},engine_mode="real")
    assert len(state["rows"])==2
    assert events==[("analyze",40),("roll_estimate",40),("analyze",40),
                    ("analyze",80),("roll_estimate",80),("analyze",80)]
    assert state["rows"][0]["errors"]["analyze_after"].startswith("RuntimeError:")
    assert state["rows"][1]["result_after"]=={"faces":[]}


def test_roll_below_threshold_reuses_before_result_without_second_analyze():
    from types import SimpleNamespace
    from web import frontalize_experiment as experiment

    crop=Image.new("RGB",(24,32),(40,20,30))
    views=SimpleNamespace(faces=[SimpleNamespace(face_id=1,detection=SimpleNamespace(cropped_face=crop))])
    before={"faces":[{"marker":"before"}]}; calls=[]
    class Frontalizer:
        def correct_roll(self,image,threshold_deg):
            return SimpleNamespace(image=image,roll_deg=4.,yaw_deg=12.,pitch_deg=-3.,applied=False,
                                   rotation_deg=0.,threshold_deg=threshold_deg,time_ms=1.)
    def analyze(data,**kwargs):
        calls.append(data)
        return before
    with patch("services.frontalization.get_tddfa_v2_frontalizer",return_value=Frontalizer()), \
         patch("web.orchestrator.analyze",side_effect=analyze):
        state=experiment.run_frontalization_experiment(views,emoca={"mode":"real"},engine_mode="real")
    row=state["rows"][0]
    assert len(calls)==1 and row["result_after"] is row["result_before"] is before
    assert row["skipped_reason"]=="roll_below_threshold" and row["pose"]["applied"] is False
