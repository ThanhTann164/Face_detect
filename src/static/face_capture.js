/**
 * Face Capture JavaScript - Thu thập ảnh tối ưu với quality check
 * Tự động lọc ảnh blur, trùng lặp, chất lượng thấp
 */

class FaceCaptureManager {
    constructor(apiBase = 'http://localhost:8000') {
        this.apiBase = apiBase;
        this.capturedImages = [];
        this.isCapturing = false;
        this.captureInterval = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        
        // Quality settings
        this.minInterval = 500; // ms giữa các lần capture
        this.targetCount = 100;
        this.qualityThreshold = 0.5;
        
        // Stats
        this.stats = {
            total: 0,
            success: 0,
            failed: 0,
            reasons: {}
        };
    }
    
    /**
     * Khởi tạo camera
     */
    async initCamera(videoElement) {
        try {
            this.video = videoElement;
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = stream;
            await this.video.play();
            
            // Tạo canvas
            this.canvas = document.createElement('canvas');
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx = this.canvas.getContext('2d');
            
            return true;
        } catch (error) {
            console.error('Camera initialization failed:', error);
            return false;
        }
    }
    
    /**
     * Capture một frame từ video
     */
    captureFrame() {
        if (!this.video || !this.canvas || !this.ctx) {
            return null;
        }
        
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        return this.canvas.toDataURL('image/jpeg', 0.92);
    }
    
    /**
     * Capture và gửi lên server với quality check
     */
    async captureWithQualityCheck(personName = '') {
        const base64Image = this.captureFrame();
        if (!base64Image) {
            return {
                success: false,
                message: 'Failed to capture frame'
            };
        }
        
        try {
            const response = await fetch(`${this.apiBase}/api/capture`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64Image,
                    person_name: personName
                })
            });
            
            const result = await response.json();
            
            // Update stats
            this.stats.total++;
            if (result.success) {
                this.stats.success++;
                this.capturedImages.push({
                    image: base64Image,
                    quality: result.quality_score,
                    timestamp: new Date().toISOString()
                });
            } else {
                this.stats.failed++;
                const reason = result.message || 'Unknown';
                this.stats.reasons[reason] = (this.stats.reasons[reason] || 0) + 1;
            }
            
            return result;
        } catch (error) {
            this.stats.total++;
            this.stats.failed++;
            return {
                success: false,
                message: `Network error: ${error.message}`
            };
        }
    }
    
    /**
     * Bắt đầu capture tự động
     */
    async startAutoCapture(personName, targetCount = 100, onProgress = null) {
        if (this.isCapturing) {
            return { success: false, message: 'Already capturing' };
        }
        
        this.isCapturing = true;
        this.targetCount = targetCount;
        this.capturedImages = [];
        this.stats = { total: 0, success: 0, failed: 0, reasons: {} };
        
        let lastCaptureTime = 0;
        
        return new Promise((resolve) => {
            const captureLoop = async () => {
                if (!this.isCapturing || this.capturedImages.length >= targetCount) {
                    this.isCapturing = false;
                    if (this.captureInterval) {
                        clearInterval(this.captureInterval);
                        this.captureInterval = null;
                    }
                    
                    resolve({
                        success: true,
                        message: `Captured ${this.capturedImages.length} images`,
                        stats: this.stats,
                        images: this.capturedImages
                    });
                    return;
                }
                
                const currentTime = Date.now();
                if (currentTime - lastCaptureTime < this.minInterval) {
                    return; // Skip nếu quá sớm
                }
                
                lastCaptureTime = currentTime;
                
                // Capture với quality check
                const result = await this.captureWithQualityCheck(personName);
                
                // Callback progress
                if (onProgress) {
                    onProgress({
                        captured: this.capturedImages.length,
                        target: targetCount,
                        success: result.success,
                        message: result.message,
                        stats: { ...this.stats }
                    });
                }
            };
            
            // Chạy capture loop mỗi minInterval ms
            this.captureInterval = setInterval(captureLoop, this.minInterval);
            captureLoop(); // Chạy ngay lần đầu
        });
    }
    
    /**
     * Dừng capture tự động
     */
    stopAutoCapture() {
        this.isCapturing = false;
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }
    }
    
    /**
     * Lưu các ảnh đã capture lên server
     */
    async saveCapturedImages(personName) {
        if (this.capturedImages.length === 0) {
            return { success: false, message: 'No images to save' };
        }
        
        try {
            const response = await fetch(`${this.apiBase}/api/save_captured_images`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    person_name: personName,
                    images: this.capturedImages.map(img => img.image)
                })
            });
            
            return await response.json();
        } catch (error) {
            return {
                success: false,
                message: `Failed to save: ${error.message}`
            };
        }
    }
    
    /**
     * Lấy thống kê
     */
    getStats() {
        return { ...this.stats };
    }
    
    /**
     * Reset stats và images
     */
    reset() {
        this.capturedImages = [];
        this.stats = { total: 0, success: 0, failed: 0, reasons: {} };
    }
    
    /**
     * Dừng camera
     */
    stopCamera() {
        if (this.video && this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        this.stopAutoCapture();
    }
}

// Export cho sử dụng trong HTML
if (typeof window !== 'undefined') {
    window.FaceCaptureManager = FaceCaptureManager;
}

