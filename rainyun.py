import logging
import os
import random
import re
import time
import subprocess
import sys

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置环境变量以避免外部服务调用
os.environ['WDM_AUTO_UPDATE'] = '0'
os.environ['WDM_SSL_VERIFY'] = '0'
os.environ['WDM_LOG_LEVEL'] = '0'
os.environ['WDM_LOCAL'] = '1'

import cv2
import ddddocr
import requests
from selenium.common import TimeoutException, WebDriverException, NoSuchElementException
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# 修改为正确的导入方式
try:
    from webdriver_manager.chrome import ChromeDriverManager
    # 尝试不同的ChromeType导入路径
    try:
        from webdriver_manager.core.utils import ChromeType
    except ImportError:
        try:
            from webdriver_manager.chrome import ChromeType
        except ImportError:
            # 如果找不到ChromeType，设置为None
            ChromeType = None
except ImportError:
    print("webdriver_manager未安装，将使用备用方式")
    ChromeDriverManager = None
    ChromeType = None

def init_selenium(debug=False, headless=False):
    """初始化Selenium WebDriver，优化ChromeDriver的使用策略"""
    # 初始化日志记录器
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    logger.info("开始初始化Selenium WebDriver...")
    chrome_options = Options()
    
    # 无论什么环境都添加无头模式选项
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    if headless or is_github_actions:
        logger.info("启用无头模式")
        for option in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
            chrome_options.add_argument(option)
    
    # 添加通用选项
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--lang=zh-CN')
    
    # 增强网络稳定性选项
    chrome_options.add_argument('--disable-features=site-per-process')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--allow-insecure-localhost')
    chrome_options.add_argument('--log-level=3')
    
    # 使用固定代理IP配置
    try:
        # 优先从环境变量获取代理设置
        proxy_url = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
        
        # 如果没有环境变量代理设置，使用固定代理IP
        if not proxy_url:
            # 使用固定代理IP 115.239.234.43:7302，使用http协议
            proxy_url = "http://115.239.234.43:7302"
            logger.info("使用固定代理IP: 115.239.234.43:7302 (HTTP协议)")
        else:
            logger.info(f"使用环境变量配置的代理: {proxy_url}")
        
        # 配置代理
        chrome_options.add_argument(f'--proxy-server={proxy_url}')
        # 信任所有SSL证书，避免代理SSL问题
        chrome_options.add_argument('--ignore-certificate-errors-spki-list=*')
    except Exception as e:
        logger.error(f"代理配置出错: {e}")
        logger.warning("将使用直接连接模式")
        # 移除可能存在的代理配置
        if '--proxy-server' in str(chrome_options.arguments):
            chrome_options.arguments = [arg for arg in chrome_options.arguments if not arg.startswith('--proxy-server')]
    
    # 添加网络超时设置
    chrome_options.set_capability('timeouts', {'pageLoad': 60000, 'script': 60000, 'implicit': 30000})
    
    # 禁用不必要的安全策略以提高连接成功率
    if is_github_actions:
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
    
    if debug and not is_github_actions:
        chrome_options.add_experimental_option("detach", True)
    
    # 尝试不同的ChromeDriver使用策略
    driver = None
    
    # 策略0: 优先检查环境变量中指定的ChromeDriver路径
    try:
        chromedriver_path = os.environ.get('CHROMEDRIVER_PATH')
        if chromedriver_path and os.path.exists(chromedriver_path):
            logger.info(f"尝试使用环境变量指定的ChromeDriver路径: {chromedriver_path}")
            service = Service(chromedriver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("成功使用环境变量指定的ChromeDriver")
            return driver
    except Exception as e:
        logger.warning(f"环境变量ChromeDriver失败: {str(e)}")
    
    # 策略1: 直接使用系统路径中的ChromeDriver（最简单可靠）
    try:
        logger.info("尝试直接使用系统ChromeDriver...")
        # 不指定service，让Selenium自动查找系统路径中的ChromeDriver
        driver = webdriver.Chrome(options=chrome_options)
        logger.info("成功使用系统ChromeDriver")
        return driver
    except Exception as e:
        logger.warning(f"系统ChromeDriver失败: {str(e)}")
    
    # 策略2: 优化webdriver-manager的使用方式，使用本地模式避免外部依赖
    try:
        logger.info("尝试使用webdriver-manager (本地模式)...")
        if ChromeDriverManager:
            # 设置临时环境变量，确保不调用外部服务
            with os.environ.copy() as temp_env:
                temp_env['WDM_LOCAL'] = '1'
                temp_env['WDM_GITHUB_TOKEN'] = ''
                
                # 仅当ChromeType可用时才指定chrome_type参数
                if ChromeType and hasattr(ChromeType, 'GOOGLE'):
                    manager = ChromeDriverManager(chrome_type=ChromeType.GOOGLE)
                else:
                    # 在新版本中，可能不再需要指定ChromeType
                    manager = ChromeDriverManager()
                
                # 尝试获取驱动路径但不自动安装
                try:
                    driver_path = manager.install()
                    logger.info(f"获取到ChromeDriver路径: {driver_path}")
                    # 手动创建service并指定正确的驱动路径
                    service = Service(driver_path)
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    logger.info("成功使用webdriver-manager")
                    return driver
                except Exception as install_error:
                    logger.warning(f"webdriver-manager安装失败: {str(install_error)}")
    except Exception as e:
        logger.warning(f"webdriver-manager失败: {str(e)}")
    
    # 策略3: 作为最后的备用，尝试使用固定路径
    try:
        logger.info("尝试使用备用ChromeDriver路径...")
        # 尝试常见的ChromeDriver路径
        common_paths = [
            '/usr/local/bin/chromedriver', '/usr/bin/chromedriver', 
            './chromedriver', 'chromedriver', 'chromedriver.exe',
            '/opt/hostedtoolcache/Python/*/x64/lib/python*/site-packages/chromedriver_binary/chromedriver'  # GitHub Actions可能的路径
        ]
        
        # 展开通配符路径
        import glob
        expanded_paths = []
        for path in common_paths:
            if '*' in path:
                expanded_paths.extend(glob.glob(path))
            else:
                expanded_paths.append(path)
        
        for path in expanded_paths:
            try:
                if os.path.exists(path):
                    logger.info(f"尝试备用路径: {path}")
                    service = Service(path)
                    # 抑制版本不匹配警告
                    service.creationflags = 0x08000000  # NoWindow
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    logger.info(f"成功使用备用路径: {path}")
                    return driver
            except Exception as path_error:
                logger.debug(f"路径 {path} 失败: {str(path_error)}")
                continue
    except Exception as e:
        logger.warning(f"备用路径搜索失败: {str(e)}")
    
    # 策略4: 在GitHub Actions环境中，尝试安装特定版本的ChromeDriver
    if is_github_actions:
        logger.info("在GitHub Actions环境中，尝试安装特定版本的ChromeDriver...")
        try:
            # 直接安装特定版本以匹配Chrome 140
            logger.info("安装chromedriver-binary==140.0.7356.93...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'chromedriver-binary==140.0.7356.93'], 
                          check=True, timeout=30)
            import chromedriver_binary  # 这个包会自动设置路径
            logger.info("导入chromedriver_binary成功")
            driver = webdriver.Chrome(options=chrome_options)
            logger.info("成功使用chromedriver-binary特定版本")
            return driver
        except Exception as e:
            logger.error(f"特定版本安装失败: {str(e)}")
    
    # 策略5: 最后尝试强制使用系统ChromeDriver，忽略版本匹配警告
    try:
        logger.info("最后尝试: 强制使用系统ChromeDriver，忽略版本警告...")
        # 创建服务时设置静默模式
        for path in ["chromedriver", "chromedriver.exe", "/usr/bin/chromedriver"]:
            try:
                service = Service(path, service_args=['--silent'])
                driver = webdriver.Chrome(service=service, options=chrome_options)
                logger.info("成功强制使用ChromeDriver")
                return driver
            except:
                continue
    except Exception as e:
        logger.error(f"强制使用失败: {str(e)}")
    
    # 所有策略都失败时的错误处理
    logger.error("无法初始化ChromeDriver，请检查Chrome和ChromeDriver的安装")
    
    # 设置超时
    if driver:
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(30)
        driver.implicitly_wait(10)
        logger.info("Selenium初始化成功！")
    else:
        logger.error("无法初始化Selenium WebDriver，请确保chromedriver已正确安装")
        raise Exception("无法初始化Selenium WebDriver")
    
    return driver

def download_image(url, filename):
    os.makedirs("temp", exist_ok=True)
    try:
        # 禁用代理以避免连接问题
        response = requests.get(url, timeout=10, proxies={"http": None, "https": None}, verify=False)
        if response.status_code == 200:
            path = os.path.join("temp", filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        else:
            logger.error(f"下载图片失败！状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"下载图片异常: {str(e)}")
        return False


def get_url_from_style(style):
    return re.search(r'url\(["\']?(.*?)["\']?\)', style).group(1)


def get_width_from_style(style):
    return re.search(r'width:\s*([\d.]+)px', style).group(1)


def get_height_from_style(style):
    return re.search(r'height:\s*([\d.]+)px', style).group(1)


def process_captcha():
    try:
        download_captcha_img()
        if check_captcha():
            logger.info("开始识别验证码")
            captcha = cv2.imread("temp/captcha.jpg")
            with open("temp/captcha.jpg", 'rb') as f:
                captcha_b = f.read()
            bboxes = det.detection(captcha_b)
            result = dict()
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                spec = captcha[y1:y2, x1:x2]
                cv2.imwrite(f"temp/spec_{i + 1}.jpg", spec)
                for j in range(3):
                    similarity, matched = compute_similarity(f"temp/sprite_{j + 1}.jpg", f"temp/spec_{i + 1}.jpg")
                    similarity_key = f"sprite_{j + 1}.similarity"
                    position_key = f"sprite_{j + 1}.position"
                    if similarity_key in result.keys():
                        if float(result[similarity_key]) < similarity:
                            result[similarity_key] = similarity
                            result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
                    else:
                        result[similarity_key] = similarity
                        result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
            if check_answer(result):
                for i in range(3):
                    similarity_key = f"sprite_{i + 1}.similarity"
                    position_key = f"sprite_{i + 1}.position"
                    positon = result[position_key]
                    logger.info(f"图案 {i + 1} 位于 ({positon})，匹配率：{result[similarity_key]}")
                    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
                    style = slideBg.get_attribute("style")
                    x, y = int(positon.split(",")[0]), int(positon.split(",")[1])
                    width_raw, height_raw = captcha.shape[1], captcha.shape[0]
                    width, height = float(get_width_from_style(style)), float(get_height_from_style(style))
                    x_offset, y_offset = float(-width / 2), float(-height / 2)
                    final_x, final_y = int(x_offset + x / width_raw * width), int(y_offset + y / height_raw * height)
                    ActionChains(driver).move_to_element_with_offset(slideBg, final_x, final_y).click().perform()
                confirm = wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="tcStatus"]/div[2]/div[2]/div/div')))
                logger.info("提交验证码")
                confirm.click()
                time.sleep(5)
                result = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="tcOperation"]')))
                if result.get_attribute("class") == 'tc-opera pointer show-success':
                    logger.info("验证码通过")
                    return
                else:
                    logger.error("验证码未通过，正在重试")
            else:
                logger.error("验证码识别失败，正在重试")
        else:
            logger.error("当前验证码识别率低，尝试刷新")
        reload = driver.find_element(By.XPATH, '//*[@id="reload"]')
        time.sleep(5)
        reload.click()
        time.sleep(5)
        process_captcha()
    except TimeoutException:
        logger.error("获取验证码图片失败")


def download_captcha_img():
    if os.path.exists("temp"):
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
    img1_style = slideBg.get_attribute("style")
    img1_url = get_url_from_style(img1_style)
    logger.info("开始下载验证码图片(1): " + img1_url)
    download_image(img1_url, "captcha.jpg")
    sprite = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="instruction"]/div/img')))
    img2_url = sprite.get_attribute("src")
    logger.info("开始下载验证码图片(2): " + img2_url)
    download_image(img2_url, "sprite.jpg")


def check_captcha() -> bool:
    """改进的验证码检查函数"""
    try:
        raw = cv2.imread("temp/sprite.jpg")
        if raw is None:
            logger.error("无法读取验证码图片")
            return False
        
        # 图像质量检查
        h, w = raw.shape[:2]
        if h < 50 or w < 100:  # 检查图像尺寸是否合理
            logger.warning(f"验证码图片尺寸过小: {w}x{h}")
            return False
        
        # 图像清晰度检查
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian < 50:  # 低于阈值可能是模糊图像
            logger.warning(f"验证码图片清晰度不足: {laplacian}")
            return False
            
        # 分割和保存三个子图像
        for i in range(3):
            w_segment = w // 3
            # 添加一定的边界裕度，避免分割到边缘
            start_x = max(0, w_segment * i + 2)
            end_x = min(w, w_segment * (i + 1) - 2)
            temp = raw[:, start_x:end_x]
            cv2.imwrite(f"temp/sprite_{i + 1}.jpg", temp)
            
            # 图像识别检查
            with open(f"temp/sprite_{i + 1}.jpg", mode="rb") as f:
                temp_rb = f.read()
            try:
                result = ocr.classification(temp_rb)
                if result in ["0", "1"]:
                    logger.warning(f"发现无效验证码: sprite_{i + 1}.jpg = {result}")
                    return False
            except Exception as e:
                logger.error(f"OCR识别出错: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"验证码检查失败: {e}")
        return False


# 检查是否存在重复坐标，快速判断识别错误
def check_answer(d: dict) -> bool:
    """改进的答案检查函数，不仅检查重复坐标，还检查相似度阈值"""
    # 检查是否有重复值
    flipped = dict()
    for key in d.keys():
        flipped[d[key]] = key
    
    if len(d.values()) != len(flipped.keys()):
        return False
    
    # 检查相似度是否达到最低阈值
    min_similarity_threshold = 0.3  # 设置最低相似度阈值
    for i in range(3):
        similarity_key = f"sprite_{i + 1}.similarity"
        if similarity_key in d and float(d[similarity_key]) < min_similarity_threshold:
            logger.warning(f"相似度不足: {similarity_key} = {d[similarity_key]}")
            return False
    
    # 检查位置是否合理分布（避免太集中）
    positions = []
    for i in range(3):
        position_key = f"sprite_{i + 1}.position"
        if position_key in d:
            x, y = map(int, d[position_key].split(","))
            positions.append((x, y))
    
    # 检查位置分布是否合理
    if len(positions) == 3:
        # 计算x坐标的分布范围
        x_coords = [p[0] for p in positions]
        x_range = max(x_coords) - min(x_coords)
        
        # 如果三个点太集中，可能识别有误
        if x_range < 50:  # 假设阈值为50像素
            logger.warning(f"位置分布过于集中: x范围 = {x_range}")
            return False
    
    return True


def preprocess_image(image):
    """图像预处理函数，提高特征匹配准确率"""
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 自适应阈值二值化，增强对比度
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 形态学操作，增强特征
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def compute_similarity(img1_path, img2_path):
    """优化的相似度计算函数"""
    # 读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # 图像尺寸标准化
    if img1.shape[0] > 100 or img1.shape[1] > 100:
        # 如果图像太大，先缩小以提高处理速度
        scale = 100.0 / max(img1.shape)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    if img2.shape[0] > 100 or img2.shape[1] > 100:
        scale = 100.0 / max(img2.shape)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # 图像预处理
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    # 使用SIFT特征提取
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0.0, 0

        # 使用FLANN匹配器，比BFMatcher更高效
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 增加检查次数以提高准确率
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 应用比例测试筛选好的匹配点，使用更严格的阈值0.7
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) == 0:
            return 0.0, 0

        # 计算相似度时考虑特征点总数和匹配点比例
        # 同时考虑特征点数量的影响，避免小图像的匹配点少但比例高的问题
        feature_factor = min(1.0, len(kp1) / 100.0, len(kp2) / 100.0)  # 归一化特征点数量因子
        match_ratio = len(good) / min(len(des1), len(des2))  # 使用最小特征点数作为分母更合理
        
        # 综合相似度计算
        similarity = match_ratio * 0.7 + feature_factor * 0.3
        
        return similarity, len(good)
    except Exception as e:
        logger.error(f"相似度计算出错: {e}")
        return 0.0, 0


# 实现main函数，添加重试机制并优化错误处理
def main(debug=False):
    """主函数，执行雨云自动签到流程"""
    # 记录开始时间
    start_time = time.time()
    
    # 连接超时等待
    timeout = 15

    # 定义环境变量
    user = os.environ.get("RAINYUN_USER", "")
    pwd = os.environ.get("RAINYUN_PASS", "")
    user = user.strip()
    pwd = pwd.strip()
    
    # 确保有用户名和密码
    if not user or not pwd:
        print("错误: 未设置用户名或密码，请在环境变量中设置RAINYUN_USER和RAINYUN_PASS")
        exit(1)
    
    # 环境变量判断是否在GitHub Actions中运行
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    # 从环境变量读取模式设置
    if not debug:  # 只有传入的debug为False时才从环境变量读取
        debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    headless = os.environ.get('HEADLESS', 'false').lower() == 'true'
    
    # 如果在GitHub Actions环境中，强制使用无头模式
    if is_github_actions:
        headless = True
    
    # 随机延时等待，避免被检测为脚本
    if not debug:
        random_sleep = random.randint(1, 10)
        logger.info(f"随机延时等待 {random_sleep} 秒")
        time.sleep(random_sleep)
    
    # 初始化 ddddocr，添加错误处理
    logger.info("初始化 ddddocr")
    try:
        ocr = ddddocr.DdddOcr(ocr=True, show_ad=False)
        det = ddddocr.DdddOcr(det=True, show_ad=False)
    except Exception as e:
        logger.error(f"初始化ddddocr失败: {e}")
        logger.warning("将尝试继续执行，但可能无法处理验证码")
        ocr = None
        det = None
    
    # 初始化 Selenium，添加重试机制
    logger.info("初始化 Selenium")
    max_retries = 3
    retry_count = 0
    driver = None
    
    while retry_count < max_retries:
        try:
            driver = init_selenium(debug=debug, headless=headless)
            break  # 成功初始化，跳出循环
        except Exception as e:
            retry_count += 1
            logger.error(f"初始化Selenium失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count < max_retries:
                # 指数退避策略
                wait_time = 2 ** retry_count + random.uniform(0, 1)
                logger.info(f"{wait_time:.2f}秒后重试...")
                time.sleep(wait_time)
            else:
                logger.critical("达到最大重试次数，无法初始化Selenium")
                raise Exception(f"无法初始化Selenium WebDriver，错误: {str(e)}")
    
    try:
        # 继续执行原有的功能
        # 过 Selenium 检测
        with open("stealth.min.js", mode="r") as f:
            js = f.read()
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": js
        })
        logger.info("发起登录请求")
        # 添加连接重试机制
        max_connection_retries = 3
        connection_retry_count = 0
        
        while connection_retry_count < max_connection_retries:
            try:
                logger.info(f"尝试连接雨云登录页面 (第{connection_retry_count + 1}/{max_connection_retries}次)")
                # 设置页面加载超时
                driver.set_page_load_timeout(30)
                driver.get("https://app.rainyun.com/auth/login")
                # 简单检查页面是否加载成功
                current_url = driver.current_url
                logger.info(f"连接成功，当前URL: {current_url}")
                break
            except Exception as e:
                connection_retry_count += 1
                logger.error(f"连接失败: {str(e)}")
                if connection_retry_count < max_connection_retries:
                    wait_time = (connection_retry_count * 2) + random.randint(1, 3)
                    logger.info(f"{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("达到最大重试次数，连接失败！")
                    raise
        
        wait = WebDriverWait(driver, timeout)
        # 改进的登录逻辑，添加重试机制
        max_retries = 3
        retry_count = 0
        login_success = False
        
        while retry_count < max_retries and not login_success:
            try:
                # 使用更可靠的定位方式
                username = wait.until(EC.visibility_of_element_located((By.NAME, 'login-field')))
                password = wait.until(EC.visibility_of_element_located((By.NAME, 'login-password')))
                
                # 尝试多种方式定位登录按钮
                try:
                    login_button = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                                        '//*[@id="app"]/div[1]/div[1]/div/div[2]/fade/div/div/span/form/button')))
                except:
                    try:
                        login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
                    except:
                        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "登录")]')))
                
                # 清除可能存在的输入
                username.clear()
                password.clear()
                
                # 添加输入延迟，模拟真实用户
                username.send_keys(user)
                time.sleep(0.5)
                password.send_keys(pwd)
                time.sleep(0.5)
                
                # 使用JavaScript点击，避免元素遮挡问题
                driver.execute_script("arguments[0].click();", login_button)
                logger.info(f"登录尝试 {retry_count + 1}/{max_retries}")
                login_success = True
            except TimeoutException:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"登录失败，{retry_count}秒后重试...")
                    time.sleep(retry_count)
                    driver.refresh()
                else:
                    logger.error("页面加载超时，请尝试延长超时时间或切换到国内网络环境！")
                    exit()
        try:
            login_captcha = wait.until(EC.visibility_of_element_located((By.ID, 'tcaptcha_iframe_dy')))
            logger.warning("触发验证码！")
            driver.switch_to.frame("tcaptcha_iframe_dy")
            process_captcha()
        except TimeoutException:
            logger.info("未触发验证码")
        time.sleep(5)
        driver.switch_to.default_content()
        # 验证登录状态并处理赚取积分
        if "dashboard" in driver.current_url:
            logger.info("登录成功！")
            logger.info("正在转到赚取积分页")
            
            # 尝试多次访问赚取积分页面
            for _ in range(3):
                try:
                    driver.get("https://app.rainyun.com/account/reward/earn")
                    logger.info("等待赚取积分页面加载...")
                    # 等待页面加载完成
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    time.sleep(3)  # 额外等待确保页面完全渲染
                    
                    # 使用多种策略查找赚取积分按钮
                    earn = None
                    strategies = [
                        (By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div[1]/div/span[2]/a'),
                        (By.XPATH, '//a[contains(@href, "earn") and contains(text(), "赚取")]'),
                        (By.CSS_SELECTOR, 'a[href*="earn"]'),
                        (By.XPATH, '//a[contains(@class, "earn")]')
                    ]
                    
                    for by, selector in strategies:
                        try:
                            earn = wait.until(EC.element_to_be_clickable((by, selector)))
                            logger.info(f"使用策略 {by}={selector} 找到赚取积分按钮")
                            break
                        except:
                            logger.debug(f"策略 {by}={selector} 未找到按钮，尝试下一种")
                            continue
                    
                    if earn:
                        # 滚动到元素位置
                        driver.execute_script("arguments[0].scrollIntoView(true);", earn)
                        time.sleep(1)
                        # 使用JavaScript点击
                        logger.info("点击赚取积分")
                        driver.execute_script("arguments[0].click();", earn)
                        
                        # 处理可能出现的验证码
                        try:
                            logger.info("检查是否需要验证码")
                            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
                            logger.info("处理验证码")
                            process_captcha()
                            driver.switch_to.default_content()
                        except:
                            logger.info("未触发验证码或验证码框架加载失败")
                            driver.switch_to.default_content()
                        
                        logger.info("赚取积分操作完成")
                        break
                    else:
                        logger.warning("未找到赚取积分按钮，刷新页面重试...")
                        driver.refresh()
                        time.sleep(3)
                except Exception as e:
                    logger.error(f"访问赚取积分页面时出错: {e}")
                    time.sleep(3)
            else:
                logger.error("多次尝试后仍无法找到赚取积分按钮")
            driver.implicitly_wait(5)
            points_raw = driver.find_element(By.XPATH,
                                             '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute(
                "textContent")
            current_points = int(''.join(re.findall(r'\d+', points_raw)))
            logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
            logger.info("任务执行成功！")
        else:
            logger.error("登录失败！")
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise
    finally:
        # 确保关闭浏览器
        if driver:
            try:
                driver.quit()
                logger.info("已关闭浏览器")
            except:
                pass
        
        # 打印总执行时间
        end_time = time.time()
        logger.info(f"任务总执行时间: {(end_time - start_time):.2f} 秒")

# 主程序入口
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ver = "2.2"
    logger.info("------------------------------------------------------------------")
    logger.info(f"雨云自动签到工作流 v{ver} by 筱序二十 ~")
    logger.info("Github发布页: https://github.com/scfcn/Rainyun-Qiandao")
    logger.info("------------------------------------------------------------------")
    
    # 从环境变量获取debug模式
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    # 执行主函数
    try:
        main(debug=debug_mode)
    except Exception as e:
        logger.critical(f"程序异常退出: {str(e)}")
        sys.exit(1)
