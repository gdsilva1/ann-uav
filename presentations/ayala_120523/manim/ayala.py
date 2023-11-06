from manim import *
from manim_slides import Slide

Text.set_default(font='Latin Modern Roman')

class Ayala(Slide):
    
    def construct(self):
        
        title = VGroup(
            Text('Redes Neurais Artificiais para'),
            Text('Detecção de Danos em Trilhos')
        ).arrange(DOWN)

        self.play(FadeIn(title))

        self.next_slide()
        self.clear()

        titulo_slide = Text(
            'Motivação'
        ).to_corner(UP)

        motivacao = BulletedList(
            'Detecção de danos em trilhos',
            'Monitoramento da Integridade Estrutural (SHM)',
            'Manutenção preditiva', tex_environment='flushleft'
        ).arrange(DOWN).to_edge(LEFT)

        objetivo = Text('OBJETIVO: Desenvolver uma IA para detectar danos nos trilhos', font_size=24).to_corner(DOWN)

        self.play(FadeIn(titulo_slide))

        self.play(FadeIn(motivacao))

        self.next_slide()

        self.play(GrowFromCenter(objetivo))

