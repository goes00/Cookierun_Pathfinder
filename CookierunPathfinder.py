import cv2
import numpy as np

# 젤리 이미지 파일 로드
def load_jelly_images(jelly_files):
    jelly_images = {}
    for color, file in jelly_files.items():
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if image is not None:
            jelly_images[color] = image
    return jelly_images

# 이미지 위치 감지
def detect_image_position(game_image, target_images, threshold=0.633):
    # 이미지를 그레이스케일로 변환
    game_image_gray = cv2.cvtColor(game_image, cv2.COLOR_BGR2GRAY)
    
    positions = []
    for target_image in target_images:
        target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(game_image_gray, target_image_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        
        for pt in zip(*loc[::-1]):
            positions.append((pt[0], pt[1], target_image.shape[1], target_image.shape[0]))

    return positions

# 젤리 위치 감지
def detect_jelly_positions(game_image, jelly_images):
    jelly_positions = {}
    for color, jelly_image in jelly_images.items():
        positions = detect_image_position(game_image, [jelly_image])
        jelly_positions[color] = positions
    return jelly_positions

# 최적 경로 찾기
def find_nearest_jelly(current_position, jelly_positions):
    min_distance = float('inf')
    nearest_jelly = None

    for color, positions in jelly_positions.items():
        for (x, y, w, h) in positions:
            # 현재 위치에서 오른쪽으로만 이동하도록 수정
            if x >= current_position[0]:
                distance = np.sqrt((current_position[0] - x) ** 2 + (current_position[1] - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_jelly = (color, (x, y, w, h))
    
    return nearest_jelly

# 최적 경로 찾기 및 시각화
def find_optimal_path(game_image, jelly_positions, initial_position):
    current_position = initial_position
    path = []

    while jelly_positions:
        nearest_jelly = find_nearest_jelly(current_position, jelly_positions)
        if nearest_jelly is None:
            break

        color, position = nearest_jelly
        path.append((color, position))
        jelly_positions[color].remove(position)
        current_position = (position[0], position[1])
        x,y,_,_ = path[0][1]
        cv2.line(game_image, initial_position, (x,y), (0, 0,255), 3)
    for i in range(len(path)-1):
        x1, y1, _, _ = path[i][1]
        x2, y2, _, _ = path[i+1][1]
        cv2.line(game_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return game_image, path


# 메인 함수
def main():
    jelly_files = {
        'silvercoin': 'silvercoin.png',
        'kingbearjelly': 'kingbearjelly.webp',
        'jelly': 'jelly.webp',
        'yellowbearjelly': 'yellowbearjelly.webp',
        'pinkbearjelly': 'pinkbearjelly.png'
    }

    game_image_file = 'cookie_run_screenshot.jpg'
    
    character_files = ['strawberryshortcakecookie.png', 'strawberryshortcakecookie1.png', 'strawberryshortcakecookie2.png', 'strawberryshortcakecookie3.png', 'strawberryshortcakecookie4.png']

    jelly_images = load_jelly_images(jelly_files)

    game_image = cv2.imread(game_image_file)
    height, width = game_image.shape[:2]

    if width != 1480 or height != 720:
        game_image = cv2.resize(game_image, (1480,720))
    if game_image is None:
        print("Failed to load game image.")
        return
    
    # 캐릭터 이미지 로드
    character_position = None
    for character_file in character_files:
        character_image = cv2.imread(character_file, cv2.IMREAD_UNCHANGED)
        if character_image is not None:
            character_position = detect_image_position(game_image, [character_image])
            if character_position:
                break
    
    if character_position is None:
        print("Character not found.")
        return
    
    initial_position = (character_position[0][0], character_position[0][1])

    # 젤리 위치 감지
    jelly_positions = detect_jelly_positions(game_image, jelly_images)

    # 최적 경로 찾기 및 시각화
    optimal_image, path = find_optimal_path(game_image, jelly_positions, initial_position)

    print("Pathfinder Result:", path)
    cv2.imshow('CR Pathfinder', optimal_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

