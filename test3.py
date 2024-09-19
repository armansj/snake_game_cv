import cv2
import numpy as np
import mediapipe as mp
import random
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

snake = [(400, 400)]
snake_radius = 15
food_size = 30
direction = (0, -1)
score = 0
game_over = False

food_pos = (random.randint(50, 750), random.randint(50, 750))

button_pos = (350, 400)
button_size = (300, 75)
button_text = "Start Again"

movement_threshold = 10

buffer_size = 10
position_buffer = deque(maxlen=buffer_size)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_snake_and_food(frame, snake, food_pos):
    for pos in snake:
        cv2.circle(frame, pos, snake_radius, (0, 255, 0), -1)

    # Draw food
    cv2.rectangle(frame, (food_pos[0] - food_size // 2, food_pos[1] - food_size // 2),
                  (food_pos[0] + food_size // 2, food_pos[1] + food_size // 2), (0, 0, 255), -1)


def move_snake(snake, direction):
    new_head = (snake[0][0] + direction[0] * snake_radius * 2, snake[0][1] + direction[1] * snake_radius * 2)
    snake = [new_head] + snake[:-1]
    return snake


def grow_snake(snake, direction):
    new_head = (snake[0][0] + direction[0] * snake_radius * 2, snake[0][1] + direction[1] * snake_radius * 2)
    snake = [new_head] + snake
    return snake


def check_collision(snake, food_pos):
    head = snake[0]
    food_x, food_y = food_pos
    head_x, head_y = head

    if (food_x - food_size // 2 <= head_x <= food_x + food_size // 2 and
            food_y - food_size // 2 <= head_y <= food_y + food_size // 2):
        return True
    return False


def check_self_collision(snake):
    head = snake[0]
    return head in snake[1:]


def check_border_collision(snake, width, height):
    head = snake[0]
    return head[0] < 0 or head[0] >= width or head[1] < 0 or head[1] >= height


def draw_game_over(frame, score):
    cv2.putText(frame, "Game Over", (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.putText(frame, f"Score: {score}", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (button_pos[0], button_pos[1]),
                  (button_pos[0] + button_size[0], button_pos[1] + button_size[1]), (0, 255, 0), -1)
    cv2.putText(frame, button_text, (button_pos[0] + 50, button_pos[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def restart_game():
    global snake, direction, score, food_pos, game_over
    snake = [(400, 400)]
    direction = (0, -1)
    score = 0
    food_pos = (random.randint(50, 750), random.randint(50, 750))
    game_over = False


def draw_direction_indicator(frame, x, y, direction):
    arrow = ""
    if direction == (-1, 0):  # Left
        arrow = "<"
    elif direction == (1, 0):  # Right
        arrow = ">"
    elif direction == (0, -1):  # Up
        arrow = "^"
    elif direction == (0, 1):  # Down
        arrow = "v"

    cv2.putText(frame, arrow, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)


def get_hand_direction(landmarks, frame_width, frame_height):
    wrist_x = int(landmarks.landmark[0].x * frame_width)
    wrist_y = int(landmarks.landmark[0].y * frame_height)

    # Averaging to smooth movement
    position_buffer.append((wrist_x, wrist_y))
    avg_x = int(np.mean([p[0] for p in position_buffer]))
    avg_y = int(np.mean([p[1] for p in position_buffer]))

    if avg_y < frame_height / 3:
        return (0, 1)  # Down
    elif avg_y > 2 * frame_height / 3:
        return (0, -1)  # Up
    elif avg_x < frame_width / 3:
        return (1, 0)  # Right
    elif avg_x > 2 * frame_width / 3:
        return (-1, 0)  # Left
    return direction


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            direction = get_hand_direction(hand_landmarks, w, h)

            wrist_x = int(hand_landmarks.landmark[0].x * w)
            wrist_y = int(hand_landmarks.landmark[0].y * h)
            draw_direction_indicator(frame, wrist_x, wrist_y, direction)

            if game_over:
                if (button_pos[0] < wrist_x < button_pos[0] + button_size[0] and
                        button_pos[1] < wrist_y < button_pos[1] + button_size[1]):
                    restart_game()
            else:
                snake = move_snake(snake, direction)

                if check_collision(snake, food_pos):
                    score += 1
                    snake = grow_snake(snake, direction)
                    food_pos = (random.randint(50, 750), random.randint(50, 750))

                if check_self_collision(snake) or check_border_collision(snake, w, h):
                    game_over = True

                draw_snake_and_food(frame, snake, food_pos)

                cv2.putText(frame, f"Score: {score}", (w // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        if game_over:
            draw_game_over(frame, score)

    if game_over:
        draw_game_over(frame, score)

    cv2.imshow("Snake Game", frame)

    # Slowing down the game speed
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
