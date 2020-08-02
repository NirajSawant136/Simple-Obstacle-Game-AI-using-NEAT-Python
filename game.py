import pygame
import time
import neat
import random
import os

pygame.font.init()

RED = (200, 30, 40)
LIGHT_RED = (240, 50, 60)
WHITE = (255, 255, 255)

WIDTH = 800
HEIGHT = 500

offset = 30

BLOCK_NOS = 7

OBSTACLE_WIDTH = 105
GAP = (WIDTH - BLOCK_NOS*OBSTACLE_WIDTH) / (BLOCK_NOS + 1)
PLAYER_SIZE = 30

CHOICE = [x for x in range(BLOCK_NOS)]

PLAYER_POS = [x for x in range(0 + int(GAP)//2, WIDTH - int(GAP)//2 - PLAYER_SIZE)]

STAT_FONT = pygame.font.SysFont("comicsans", 50) 

class Player:
	def __init__(self):
		self.s = PLAYER_SIZE
		self.x = HEIGHT - GAP/2 - PLAYER_SIZE
		self.y = random.choice(PLAYER_POS)

	def draw(self, win):
		pygame.draw.rect(win, WHITE, (self.y, self.x, self.s, self.s), 0)
		pygame.display.update()

	def move(self, command):
		self.y = self.y + command*(self.s // 2)

		if self.y <= 0:
			self.y = 0

		if self.y >= WIDTH:
			self.y = WIDTH - self.s

class Obstacle:
	def __init__(self):
		self.x = 0
		self.y = 0
		self.vel = 2
		self.BLOCK_LIST = [((x+1)*GAP + x*OBSTACLE_WIDTH, self.x, OBSTACLE_WIDTH, 30) for x in range(BLOCK_NOS)]
		self.missing = random.choice(CHOICE)
		self.BLOCK_LIST.remove(self.BLOCK_LIST[self.missing])
		self.gap = self.missing*(GAP + OBSTACLE_WIDTH)

	def draw(self, win):
		for block in self.BLOCK_LIST:
			pygame.draw.rect(win, WHITE, block, 0)

		pygame.draw.rect(win, RED, (0, 0, WIDTH, HEIGHT), int(GAP))
		pygame.display.update()


	def move(self):

		self.x += self.vel
		self.BLOCK_LIST = [((x+1)*GAP + x*OBSTACLE_WIDTH, self.x, OBSTACLE_WIDTH, 30) for x in range(BLOCK_NOS)]

		self.BLOCK_LIST.remove(self.BLOCK_LIST[self.missing])

def eval_genomes(genomes, config):
	win = pygame.display.set_mode((WIDTH, HEIGHT))
	clock = pygame.time.Clock()

	nets = []
	ge = []
	players = []
	obstacles = []
	score = 0

	for _,g in genomes:
		net = neat.nn. FeedForwardNetwork.create(g, config)
		nets.append(net)
		g.fitness = 0
		ge.append(g)
		players.append(Player())

	obstacles.append(Obstacle())
	obstacles[0].x = 250

	for player in players:
		player.draw(win)

	run = True
	while run:
		# clock.tick(60)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()

		win.fill(LIGHT_RED)
		pygame.draw.rect(win, RED, (0, 0, WIDTH, HEIGHT), int(GAP))

		pygame.display.update()

		for x, player in enumerate(players):
			win.fill(LIGHT_RED)
			pygame.draw.rect(win, RED, (0, 0, WIDTH, HEIGHT), int(GAP))
			# pygame.display.update()
			
			ge[x].fitness += 0.1

			index = 0
			for obstacle in obstacles:

				obstacle.move()
				obstacle.draw(win)

				if obstacle.x >= 370:
					index = obstacles.index(obstacle)

			inputs = ((player.x - obstacles[index].x), (obstacles[index].gap + 2*GAP + OBSTACLE_WIDTH - PLAYER_SIZE - player.y), (player.y + PLAYER_SIZE - obstacles[index].gap))

			output = nets[x].activate(inputs)

			finalOP = output.index(max(output)) - 1

			player.move(finalOP)
			player.draw(win)

			if player.x <= obstacles[index].x:
				ge[x].fitness += 10


			#collision 
			# if ge[x].fitness >= 10000:
			# 	run = False
			# 	break

			rem = []
			if obstacles[index].x + 30 >= player.x:
				if player.y > obstacles[index].gap + 2*GAP + OBSTACLE_WIDTH - PLAYER_SIZE:
					ge[x].fitness -= 5
					nets.pop(x)
					ge.pop(x)
					players.remove(player)

				elif player.y <= obstacles[index].gap:
					ge[x].fitness -= 5
					nets.pop(x)
					ge.pop(x)
					players.remove(player)

			
			
			for obstacle in obstacles:
				if obstacle.x >= 270 and obstacle.x < HEIGHT:
					if len(obstacles) == 1:
						obstacles.append(Obstacle())

				elif obstacle.x > HEIGHT:
					score += 1

					text = STAT_FONT.render(str(score), 1, (255,255,255))
					win.blit(text, (WIDTH//2 - text.get_width()//2, 10))
					pygame.display.update()

					obstacles.remove(obstacle)


			
		text = STAT_FONT.render(str(score), 1, (255,255,255))
		win.blit(text, (WIDTH//2 - text.get_width()//2, 10))
		pygame.display.update()

		if len(players) == 0:
			run = False


def run(config_path):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
						neat.DefaultSpeciesSet, neat.DefaultStagnation,
						config_path)

	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winner = p.run(eval_genomes)

	print("Best fitness -> {}".format(winner))


if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-FeedForward.txt")
	run(config_path)


# FINAL NETWORK ->

# Nodes:
# 	0 DefaultNodeGene(key=0, bias=-0.9866681151537819, response=1.0, activation=tanh, aggregation=sum)
# 	1 DefaultNodeGene(key=1, bias=-0.19684707599816334, response=1.0, activation=tanh, aggregation=sum)
# 	2 DefaultNodeGene(key=2, bias=-0.023412587732523353, response=1.0, activation=tanh, aggregation=sum)
# Connections:
# 	DefaultConnectionGene(key=(-3, 0), weight=0.413094212700832, enabled=True)
# 	DefaultConnectionGene(key=(-3, 1), weight=0.38493758445483867, enabled=True)
# 	DefaultConnectionGene(key=(-3, 2), weight=0.3326930938849062, enabled=True)
# 	DefaultConnectionGene(key=(-2, 0), weight=-1.4191333690437578, enabled=True)
# 	DefaultConnectionGene(key=(-2, 1), weight=-1.604381939943581, enabled=True)
# 	DefaultConnectionGene(key=(-2, 2), weight=-1.8972709882162406, enabled=True)
# 	DefaultConnectionGene(key=(-1, 1), weight=-2.030443212728862, enabled=True)
# 	DefaultConnectionGene(key=(-1, 2), weight=-1.0869417163203305, enabled=True)
# 	DefaultConnectionGene(key=(2, 2), weight=0.6586795750845641, enabled=True)

