from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
import ccxt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import asyncio

# Khởi tạo KuCoin
exchange = ccxt.kucoin()
# Lưu trữ lịch sử tín hiệu
signal_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gửi tin nhắn chào mừng và hướng dẫn."""
    await update.message.reply_text(
        "Chào mừng! Tôi là bot phân tích kỹ thuật của anh Hưng Thạnh đẹp trai.\n"
        "Gõ /chart <mã giao dịch> để xem biểu đồ kỹ thuật (ví dụ: /chart BTC/USDT).\n"
        "Gõ /top để xem top 10 cặp giao dịch tăng và giảm mạnh nhất trong 1 giờ qua.\n"
        "Gõ /signal <mã giao dịch> để nhận tín hiệu mua bán và lưu lịch sử.\n"
        "Gõ /history để xem lịch sử tín hiệu.\n"
        "Gõ /cap <mã giao dịch> để xem thông tin giá hiện tại."
    )

async def current_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hiển thị thông tin giá hiện tại của một mã giao dịch."""
    try:
        # Lấy mã giao dịch từ context.args
        symbol = context.args[0] if context.args else None
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /cap BTC/USDT")
            return

        # Lấy thông tin ticker từ KuCoin
        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return

        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        percentage_change = ticker['percentage']
        timestamp = pd.to_datetime(ticker['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')


        # Gửi thông tin giá
        message = (
            f"Thông tin giá hiện tại cho {symbol}:\n"
            f"- Giá hiện tại: {current_price:.2f} USD\n"
            f"- Biến động trong 1 giờ qua: {percentage_change:.2f}%\n"
            f"- Thời gian cập nhật: {timestamp}"
        )
        await update.message.reply_text(message)

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tạo và gửi biểu đồ kỹ thuật."""
    try:
        # Lấy mã giao dịch từ context.args hoặc callback_data
        symbol = context.args[0] if context.args else context.chat_data.get("symbol")
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /chart BTC/USDT")
            return


        timeframe = '1h'
        limit = 200


        # Kiểm tra xem mã giao dịch có hợp lệ không
        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return


        # Lấy dữ liệu từ KuCoin
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')


        # Tính toán các chỉ báo kỹ thuật
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()


        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - df['close'].rolling(window=20).std()


        # MACD
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']


        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))


        # Tạo biểu đồ
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.3, 0.2],
            specs=[[{"secondary_y": True}], [{}], [{}]]
        )


        # Candlestick và Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Candlestick"
        ), row=1, col=1, secondary_y=False)


        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Upper'],
            mode='lines',
            line=dict(color='red', width=1),
            name='BB Upper'
        ), row=1, col=1, secondary_y=False)


        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Middle'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='BB Middle'
        ), row=1, col=1, secondary_y=False)


        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Lower'],
            mode='lines',
            line=dict(color='green', width=1),
            name='BB Lower'
        ), row=1, col=1, secondary_y=False)


        # Biểu đồ khối lượng bên trục y2, cùng màu với giá
        volume_colors = [
            'green' if row['close'] > row['open'] else 'red'
            for _, row in df.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors
        ), row=1, col=1, secondary_y=True)


        # Biểu đồ MACD
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['MACD_Hist'],
            name='MACD Histogram'
        ), row=2, col=1)


        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['MACD'],
            mode='lines',
            line=dict(color='green', width=1),
            name='MACD'
        ), row=2, col=1)


        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['Signal'],
            mode='lines',
            line=dict(color='red', width=1),
            name='Signal'
        ), row=2, col=1)


        # Biểu đồ RSI
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['RSI'],
            mode='lines',
            line=dict(color='purple', width=1),
            name='RSI'
        ), row=3, col=1)


        # Đường giới hạn RSI
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[70] * len(df),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Overbought (70)'
        ), row=3, col=1)


        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[30] * len(df),
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Oversold (30)'
        ), row=3, col=1)


        # Layout
        fig.update_layout(
            title=f"{symbol} Technical Analysis Chart (1H)",
            template="plotly_dark",
            height=1000,
            xaxis_rangeslider_visible=False
        )


        # Lưu biểu đồ thành HTML
        temp_file = f"{symbol.replace('/', '_')}_chart.html"
        fig.write_html(temp_file)


        # Gửi file HTML qua Telegram
        if update.callback_query:
            with open(temp_file, 'rb') as html_file:
                await update.callback_query.message.reply_document(document=html_file, filename=temp_file)
        else:
            with open(temp_file, 'rb') as html_file:
                await update.message.reply_document(document=html_file, filename=temp_file)


        # Xóa file tạm
        os.remove(temp_file)
    except Exception as e:
        if update.callback_query:
            await update.callback_query.message.reply_text(f"Đã xảy ra lỗi: {e}")
        else:
            await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gửi danh sách top 10 cặp giao dịch tăng và giảm mạnh nhất với nút tương tác."""
    try:
        # Lấy dữ liệu thị trường từ KuCoin
        markets = exchange.fetch_tickers()
        data = []


        # Tính toán phần trăm biến động giá
        for symbol, ticker in markets.items():
            change = ticker.get('percentage')
            if change is not None:
                data.append((symbol, change))


        # Lấy top 10 tăng và giảm mạnh nhất
        top_gainers = sorted(data, key=lambda x: x[1], reverse=True)[:10]
        top_losers = sorted(data, key=lambda x: x[1])[:10]


        # Tạo danh sách nút tương tác cho top tăng
        gainers_keyboard = [
            [InlineKeyboardButton(f"{symbol}: +{change:.2f}%", callback_data=symbol)]
            for symbol, change in top_gainers
        ]


        # Tạo danh sách nút tương tác cho top giảm
        losers_keyboard = [
            [InlineKeyboardButton(f"{symbol}: {change:.2f}%", callback_data=symbol)]
            for symbol, change in top_losers
        ]


        # Gửi danh sách top tăng mạnh nhất
        await update.message.reply_text(
            "Top 10 cặp giao dịch tăng mạnh nhất trong 1 giờ qua:",
            reply_markup=InlineKeyboardMarkup(gainers_keyboard)
        )


        # Gửi danh sách top giảm mạnh nhất
        await update.message.reply_text(
            "Top 10 cặp giao dịch giảm mạnh nhất trong 1 giờ qua:",
            reply_markup=InlineKeyboardMarkup(losers_keyboard)
        )
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý nút bấm từ danh sách /top để hiển thị biểu đồ kỹ thuật."""
    query = update.callback_query
    await query.answer()


    # Lấy mã giao dịch từ callback_data
    symbol = query.data
    context.chat_data["symbol"] = symbol  # Lưu vào chat_data để gọi lại nếu cần
    await chart(update, context)






async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Phân tích và gửi tín hiệu mua bán."""
    try:
        # Lấy mã giao dịch từ context.args
        symbol = context.args[0] if context.args else None
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /signal BTC/USDT")
            return


        timeframe = '1h'
        limit = 200


        # Kiểm tra xem mã giao dịch có hợp lệ không
        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return


        # Lấy dữ liệu từ KuCoin
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')


        # Tính toán các chỉ báo kỹ thuật
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()


        # Phát hiện tín hiệu mua bán
        last_row = df.iloc[-1]  # Lấy dòng dữ liệu cuối cùng
        signals = []

        # Thời điểm và giá hiện tại
        current_time = last_row['timestamp']
        current_price = last_row['close']

        # Tín hiệu mua
        if last_row['close'] > last_row['MA50'] and last_row['MACD'] > last_row['Signal'] and last_row['RSI'] < 30:
         signals.append(f"Mua: Giá {current_price:.2f} USD vào lúc {current_time}.")
        elif last_row['close'] <= last_row['BB_Lower']:
         signals.append(f"Mua: Giá {current_price:.2f} USD vào lúc {current_time}.")

        # Tín hiệu bán
        if last_row['close'] < last_row['MA50'] and last_row['MACD'] < last_row['Signal'] and last_row['RSI'] > 70:
         signals.append(f"Bán: Giá {current_price:.2f} USD vào lúc {current_time}.")
        elif last_row['close'] >= last_row['BB_Upper']:
         signals.append(f"Bán: Giá {current_price:.2f} USD vào lúc {current_time}.")



        # Gửi tín hiệu qua Telegram và lưu vào lịch sử
        if signals:
            signal_message = f"Tín hiệu giao dịch cho {symbol}:\n" + "\n".join(signals)
            if symbol not in signal_history:
                signal_history[symbol] = []
            signal_history[symbol].append({"signals": signals, "timestamp": last_row['timestamp']})
            await update.message.reply_text(signal_message)
        else:
            await update.message.reply_text(f"Hiện tại không có tín hiệu rõ ràng cho {symbol}.")
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def monitor_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Theo dõi tín hiệu và tự động gửi thông báo."""
    try:
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # Danh sách mã giao dịch theo dõi
        timeframe = '1h'
        limit = 200


        for symbol in symbols:
            markets = exchange.load_markets()
            if symbol not in markets:
                continue


            # Lấy dữ liệu từ KuCoin
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')


            # Tính toán các chỉ báo kỹ thuật
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['EMA12'] = df['close'].ewm(span=12).mean()
            df['EMA26'] = df['close'].ewm(span=26).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9).mean()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['BB_Middle'] = df['close'].rolling(window=20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()


            # Phát hiện tín hiệu mua bán
            last_row = df.iloc[-1]
            signals = []


            # Tín hiệu mua
            if last_row['close'] > last_row['MA50'] and last_row['MACD'] > last_row['Signal'] and last_row['RSI'] < 30:
                signals.append(f"Mua: {symbol} - Giá cắt MA50, MACD tăng và RSI dưới 30.")
            elif last_row['close'] <= last_row['BB_Lower']:
                signals.append(f"Mua: {symbol} - Giá gần dải BB dưới (quá bán).")


            # Tín hiệu bán
            if last_row['close'] < last_row['MA50'] and last_row['MACD'] < last_row['Signal'] and last_row['RSI'] > 70:
                signals.append(f"Bán: {symbol} - Giá cắt MA50, MACD giảm và RSI trên 70.")
            elif last_row['close'] >= last_row['BB_Upper']:
                signals.append(f"Bán: {symbol} - Giá gần dải BB trên (quá mua).")


            # Gửi tín hiệu qua Telegram nếu có tín hiệu
            if signals:
                signal_message = f"Tín hiệu tự động cho {symbol}:\n" + "\n".join(signals)
                if symbol not in signal_history:
                    signal_history[symbol] = []
                signal_history[symbol].append({"signals": signals, "timestamp": last_row['timestamp']})
                await context.bot.send_message(chat_id=context.job.chat_id, text=signal_message)
    except Exception as e:
        print(f"Error in monitor_signals: {e}")


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hiển thị lịch sử tín hiệu giao dịch."""
    if not signal_history:
        await update.message.reply_text("Chưa có tín hiệu nào trong lịch sử.")
        return


    history_message = "Lịch sử tín hiệu giao dịch:\n"
    for symbol, entries in signal_history.items():
        history_message += f"\nMã giao dịch: {symbol}\n"
        for entry in entries[-5:]:  # Hiển thị tối đa 5 tín hiệu gần nhất cho mỗi mã giao dịch
            history_message += f"- Tín hiệu: {'; '.join(entry['signals'])}\n  Thời gian: {entry['timestamp']}\n"


    await update.message.reply_text(history_message)


def main():
    # Thay YOUR_TOKEN bằng token từ BotFather
    TOKEN = "8081244500:AAH6zjPyaYIVOpmxBK-SvJ9WPTvJ0JRcD_c"
    application = Application.builder().token(TOKEN).build()

    # Đăng ký các handler
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("chart", chart))
    application.add_handler(CommandHandler("signal", signal))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(CommandHandler("top", top))  # Thêm handler cho /top
    application.add_handler(CommandHandler("cap", current_price))  # Thêm handler cho /cap
    application.add_handler(CallbackQueryHandler(button))  # Thêm handler cho nút bấm từ /top

    # Thêm job định kỳ để giám sát tín hiệu
    job_queue = application.job_queue
    job_queue.run_repeating(monitor_signals, interval=3600, first=0)  # Kiểm tra tín hiệu mỗi giờ

    # Chạy bot
    application.run_polling()

if __name__ == "__main__":
    main()





