// Firebase configuration
const firebaseConfig = {
    // Replace with your Firebase project config
    apiKey: "your-api-key",
    authDomain: "your-project.firebaseapp.com",
    databaseURL: "https://your-project-default-rtdb.firebaseio.com",
    projectId: "your-project",
    storageBucket: "your-project.appspot.com",
    messagingSenderId: "123456789",
    appId: "your-app-id"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const database = firebase.database();

class UserCounter {
    constructor() {
        this.userId = this.generateUserId();
        this.isActive = true;
        this.heartbeatInterval = null;
        this.counterElement = document.getElementById('userCount');
        
        this.init();
    }

    generateUserId() {
        return 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    init() {
        // Add user to active users
        this.addUser();
        
        // Listen for changes in active users count
        this.listenForUserCount();
        
        // Start heartbeat to maintain presence
        this.startHeartbeat();
        
        // Handle page visibility changes
        this.handleVisibilityChange();
        
        // Remove user when page unloads
        this.handlePageUnload();
    }

    addUser() {
        const userRef = database.ref(`activeUsers/${this.userId}`);
        userRef.set({
            timestamp: firebase.database.ServerValue.TIMESTAMP,
            lastSeen: firebase.database.ServerValue.TIMESTAMP
        });
        
        // Set up automatic removal when user disconnects
        userRef.onDisconnect().remove();
    }

    listenForUserCount() {
        const activeUsersRef = database.ref('activeUsers');
        
        activeUsersRef.on('value', (snapshot) => {
            const users = snapshot.val();
            const userCount = users ? Object.keys(users).length : 0;
            this.updateCounterDisplay(userCount);
        });
    }

    updateCounterDisplay(count) {
        if (this.counterElement) {
            this.counterElement.textContent = `• ${count}`;
            
            // Add a subtle animation when count changes
            this.counterElement.style.transform = 'scale(1.1)';
            setTimeout(() => {
                this.counterElement.style.transform = 'scale(1)';
            }, 200);
        }
    }

    startHeartbeat() {
        // Update timestamp every 30 seconds to show user is still active
        this.heartbeatInterval = setInterval(() => {
            if (this.isActive) {
                const userRef = database.ref(`activeUsers/${this.userId}`);
                userRef.update({
                    lastSeen: firebase.database.ServerValue.TIMESTAMP
                });
            }
        }, 30000);
    }

    handleVisibilityChange() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.isActive = false;
                // Remove user when tab becomes hidden
                database.ref(`activeUsers/${this.userId}`).remove();
            } else {
                this.isActive = true;
                // Re-add user when tab becomes visible
                this.addUser();
            }
        });
    }

    handlePageUnload() {
        window.addEventListener('beforeunload', () => {
            // Remove user when page is about to unload
            database.ref(`activeUsers/${this.userId}`).remove();
        });
    }

    cleanup() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
        if (this.userId) {
            database.ref(`activeUsers/${this.userId}`).remove();
        }
    }
}

// Initialize user counter when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Add a small delay to ensure Firebase is loaded
    setTimeout(() => {
        window.userCounter = new UserCounter();
    }, 1000);
});

// Fallback: Clean up old users (older than 2 minutes)
setInterval(() => {
    const cutoffTime = Date.now() - (2 * 60 * 1000); // 2 minutes ago
    const activeUsersRef = database.ref('activeUsers');
    
    activeUsersRef.once('value', (snapshot) => {
        const users = snapshot.val();
        if (users) {
            Object.keys(users).forEach(userId => {
                const user = users[userId];
                if (user.lastSeen < cutoffTime) {
                    database.ref(`activeUsers/${userId}`).remove();
                }
            });
        }
    });
}, 60000); // Run every minute 