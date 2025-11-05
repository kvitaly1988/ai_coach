from __future__ import annotations
def generate_tips(angle_mae_deg, thresholds_deg=None):
    if thresholds_deg is None: thresholds_deg={'minor':8,'major':18}
    tips=[]; joint_ru={'elbow_left':'левый локоть','elbow_right':'правый локоть','shoulder_left':'левое плечо','shoulder_right':'правое плечо','hip_left':'левое бедро','hip_right':'правое бедро','knee_left':'левое колено','knee_right':'правое колено','ankle_left':'левая лодыжка','ankle_right':'правая лодыжка','torso':'корпус (наклон)'}
    for k,err in angle_mae_deg.items():
        j=joint_ru.get(k,k)
        if err>=thresholds_deg.get('major',18): tips.append(f"{j}: сильное отклонение ≈ {err:.1f}° — потренируйте амплитуду и фиксацию.")
        elif err>=thresholds_deg.get('minor',8): tips.append(f"{j}: заметное отклонение ≈ {err:.1f}° — держите угол точнее и следите за стабильностью.")
    if not tips: tips=['Техника близка к эталону — так держать! Добавьте контроль темпа и плавности.']
    return tips
